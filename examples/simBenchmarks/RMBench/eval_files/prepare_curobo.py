"""Enable cuRobo v0.7.8's LBFGS fallback in the pod-local installation.

The source patch also reaches multiprocessing planner workers. It does not
modify the shared RMBench checkout or the fused CUDA extension itself.
See https://github.com/RoboTwin-Platform/RoboTwin/issues/452.
"""

import ast
import importlib.util
from pathlib import Path


MARKER = "# starVLA: use cuRobo's unfused LBFGS implementation"


def patch_source(source):
    if MARKER in source:
        return source
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LBFGSOpt")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    # Override after config copying, before base initialization allocates buffers.
    anchor = next(
        n for n in init.body
        if isinstance(n, ast.Expr) and ast.unparse(n.value) == "NewtonOptBase.__init__(self)"
    )
    lines = source.splitlines(keepends=True)
    indent = " " * anchor.col_offset
    lines.insert(anchor.lineno - 1, f"{indent}{MARKER}\n{indent}self.use_cuda_kernel = False\n")
    result = "".join(lines)
    ast.parse(result)
    return result


def main():
    spec = importlib.util.find_spec("curobo")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("cuRobo package not found in the simulation interpreter")
    path = Path(next(iter(spec.submodule_search_locations))) / "opt/newton/lbfgs.py"
    source = path.read_text()
    patched = patch_source(source)
    if source != patched:
        path.write_text(patched)
    print(f"[cuRobo] fused LBFGS disabled: {path}", flush=True)

    # Import only after patching, so this process tests the same path as workers.
    import torch
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

    print(f"[cuRobo] warmup on {torch.cuda.get_device_name(0)}", flush=True)
    config = MotionGenConfig.load_from_robot_config(
        "franka.yml", "collision_table.yml", num_trajopt_seeds=1,
    )
    planner = MotionGen(config)
    planner.warmup()
    torch.cuda.synchronize()
    print("[cuRobo] WARMUP PASSED (unfused LBFGS)", flush=True)


if __name__ == "__main__":
    main()
