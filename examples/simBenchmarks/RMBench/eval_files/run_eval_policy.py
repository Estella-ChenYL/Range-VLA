"""Run upstream RMBench eval with unexpected setup exceptions propagated.

Transform only the handler that prints 'error occurs !', in memory. The
shared upstream checkout and its expected UnStableError retries stay intact.
"""

import ast
import os
from pathlib import Path
import sys
import time


def compile_eval(source, filename, test_num=None, output_dir=None):
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "test_num" for t in node.targets) and test_num is not None:
            node.value = ast.Constant(test_num)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "save_dir" for t in node.targets) and isinstance(node.value, ast.Call) and output_dir:
            node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()), args=[ast.Constant(output_dir)], keywords=[])
    # Upstream description generation uses Python random without seeding it.
    # Fix it at the same post-setup point in both serial and parallel modes.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "episode_info_list" for t in node.targets):
            node.value = ast.Subscript(value=ast.Tuple(elts=[
                ast.parse("__import__('random').seed(now_seed)", mode="eval").body, node.value
            ], ctx=ast.Load()), slice=ast.Constant(1), ctx=ast.Load())
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if not isinstance(node.type, ast.Name) or node.type.id != "Exception":
            continue
        if any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name) and child.func.id == "print"
            and any(isinstance(arg, ast.Constant) and arg.value == "error occurs !" for arg in child.args)
            for statement in node.body for child in ast.walk(statement)
        ):
            matches.append(node)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one RMBench setup error handler, found {len(matches)}")
    handler = matches[0]
    # No feasible grasp is a rejected expert seed, not a policy failure. Keep
    # upstream cleanup/seed advancement for this specific assertion only.
    # Everything else (including CUDA errors) still fails immediately.
    handler.name = handler.name or "_starvla_error"
    guard = ast.parse(f'''
if not (isinstance({handler.name}, AssertionError) and
        str({handler.name}) == "target_pose cannot be None for move action."):
    raise
_starvla_grasp_failures = locals().get("_starvla_grasp_failures", 0) + 1
print(f"[RMBench] rejecting seed {{now_seed}}: no feasible grasp "
      f"({{_starvla_grasp_failures}}/20 consecutive failures)", flush=True)
if _starvla_grasp_failures >= 20:
    raise RuntimeError("20 consecutive expert seeds have no feasible grasp; check planner/assets") from {handler.name}
''').body
    handler.body = guard + handler.body
    # Reset only when the expert try block completes without an exception.
    parent = next(n for n in ast.walk(tree) if isinstance(n, ast.Try) and handler in n.handlers)
    parent.body.append(ast.parse("_starvla_grasp_failures = 0").body[0])
    return compile(ast.fix_missing_locations(tree), filename, "exec")


if __name__ == "__main__":
    script = Path(sys.argv[1]).resolve()
    from parallel_eval import run
    batch_size = int(os.environ.get("BATCH_SIZE", "1"))
    test_num = int(os.environ.get("TEST_NUM") or "100")
    if batch_size < 1 or test_num < 1:
        raise ValueError("BATCH_SIZE and TEST_NUM must be positive")
    code = compile_eval(script.read_text(), str(script), test_num, os.environ.get("RMBENCH_TASK_OUTPUT"))
    sys.argv = sys.argv[1:]
    sys.path.insert(0, str(script.parent))
    # Preserve upstream path calculations and multiprocessing's main-module path.
    __file__ = str(script)
    started = time.monotonic()
    if batch_size > 1:
        run(script, code)
    else:
        exec(code, globals())
        print(f"episodes={test_num} workers=1 episodes_per_second={test_num / (time.monotonic() - started):.4f}", flush=True)
