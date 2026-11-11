"""
A restricted Python executor for math solutions: the model (or TinyGSM's data) writes a function, and the tool runs it
and returns the result as text.

The code comes from a downloaded dataset or from the model, so it's treated as untrusted:
- the syntax tree must use only plain computation (assignments, arithmetic, loops, functions, comprehensions), with
  imports limited to math and fractions and no names or attributes starting with an underscore
- it runs with a small set of builtins (no open, eval, exec, getattr or __import__) and a time limit
- PythonTool runs it in a separate process with a memory cap, so a runaway loop or allocation can't take down training
"""
import ast
import math
import multiprocessing
import signal

ALLOWED_NODES = {
    "Module", "FunctionDef", "arguments", "arg", "Return", "Assign", "AugAssign", "AnnAssign", "Expr", "Pass",
    "If", "For", "While", "Break", "Continue", "IfExp", "BoolOp", "And", "Or", "BinOp", "UnaryOp", "Compare",
    "Add", "Sub", "Mult", "Div", "FloorDiv", "Mod", "Pow", "USub", "UAdd", "Not",
    "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "In", "NotIn", "Is", "IsNot",
    "Call", "keyword", "Name", "Load", "Store", "Constant", "Tuple", "List", "Dict", "Set", "Subscript", "Slice",
    "ListComp", "SetComp", "DictComp", "GeneratorExp", "comprehension", "Attribute", "JoinedStr", "FormattedValue",
    "ImportFrom", "Import", "alias", "Lambda", "NamedExpr",
}
ALLOWED_MODULES = {"math", "fractions"}
SAFE_BUILTINS = {name: getattr(__builtins__, name) if not isinstance(__builtins__, dict) else __builtins__[name]
                 for name in ["abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "int",
                              "len", "list", "map", "max", "min", "pow", "range", "reversed", "round", "set",
                              "sorted", "str", "sum", "tuple", "zip", "isinstance", "ZeroDivisionError",
                              "ValueError", "Exception"]}
TIME_LIMIT_S = 1.0
MEMORY_LIMIT_BYTES = 1 << 30  # beyond what the worker process already uses


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    if name not in ALLOWED_MODULES:
        raise ImportError(f"import of {name} isn't allowed")
    return __import__(name, globals, locals, fromlist, level)


def check(code: str) -> ast.Module:
    """Parses the code and raises ValueError if it uses anything outside the allowed subset."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        kind = type(node).__name__
        if kind not in ALLOWED_NODES:
            raise ValueError(f"{kind} isn't allowed")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError("names starting with _ aren't allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("attributes starting with _ aren't allowed")
        if isinstance(node, ast.Import) and any(a.name not in ALLOWED_MODULES for a in node.names):
            raise ValueError("only math and fractions can be imported")
        if isinstance(node, ast.ImportFrom) and node.module not in ALLOWED_MODULES:
            raise ValueError("only math and fractions can be imported")
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and abs(node.value) > 10 ** 12:
            raise ValueError("constant too large")
    return tree


def format_result(value) -> str:
    """Numbers as the calculator formats them: integers plainly, others rounded to 4 decimals. Raises otherwise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) and not hasattr(value, "numerator"):
        raise ValueError(f"the result is a {type(value).__name__}, not a number")
    value = float(value)
    if not math.isfinite(value) or abs(value) > 1e15:
        raise ValueError("the result isn't a finite number")
    value = round(value, 4)
    return str(int(value)) if value == int(value) else f"{value:.4f}".rstrip("0")


def _timeout(signum, frame):
    raise TimeoutError("time limit exceeded")


def execute(code: str, function: str) -> str:
    """
    Runs the code, calls function() and returns its formatted result, or 'Error: ...'. Runs in the calling process,
    so use it only in a worker process with limits set (as PythonTool and prepare_sft's workers do).
    """
    try:
        tree = check(code)
        namespace = {"__builtins__": {**SAFE_BUILTINS, "__import__": _import, "print": lambda *a, **k: None}}
        signal.signal(signal.SIGALRM, _timeout)
        signal.setitimer(signal.ITIMER_REAL, TIME_LIMIT_S)
        try:
            exec(compile(tree, "<solution>", "exec"), namespace)
            if not callable(namespace.get(function)):
                raise ValueError(f"{function} isn't defined")
            result = namespace[function]()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        return format_result(result)
    except BaseException as e:  # anything the code raises, including MemoryError and RecursionError
        if isinstance(e, KeyboardInterrupt):
            raise
        return f"Error: {type(e).__name__}"


def limit_memory(extra: int = MEMORY_LIMIT_BYTES):
    """
    Caps the process's address space at its current size plus extra, so a runaway allocation fails with MemoryError.
    The cap is relative because a worker that has imported torch already maps several GB.
    """
    import resource
    with open("/proc/self/status") as f:
        current = next(int(line.split()[1]) * 1024 for line in f if line.startswith("VmSize:"))
    resource.setrlimit(resource.RLIMIT_AS, (current + extra, current + extra))


class PythonTool:
    """Runs execute() in a separate, memory-limited process, restarting it if a call hangs or the process dies."""
    def __init__(self):
        self.pool = None

    def __call__(self, code: str, function: str) -> str:
        if self.pool is None:
            self.pool = multiprocessing.get_context("forkserver").Pool(1, initializer=limit_memory)
        try:
            return self.pool.apply_async(execute, (code, function)).get(timeout=TIME_LIMIT_S + 4)
        except Exception as e:
            self.pool.terminate()
            self.pool = None
            return f"Error: {type(e).__name__}"
