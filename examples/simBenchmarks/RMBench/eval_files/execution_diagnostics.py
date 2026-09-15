"""Opt-in records of the actual RMBench TOPP calls and task failure state.

RMBench catches TOPP exceptions and keeps stepping the simulator. Capture
those exceptions without changing its fallback or success conditions.
"""

import json
from pathlib import Path

import numpy as np


def take_action_with_diagnostics(env, action, trace_dir):
    step = env.take_action_cnt
    record = {
        "step": step,
        "target_qpos": np.asarray(action).tolist(),
        "fail_flag_before": getattr(env, "fail_flag", None),
        "task_counter_before": getattr(env, "get_obs_cnt", None),
        "topp": {},
    }
    originals = []

    def observed_topp(arm, original):
        def call(path, *args, **kwargs):
            path = np.asarray(path)
            details = {"max_joint_delta": float(np.max(np.abs(path[-1] - path[0])))}
            record["topp"][arm] = details
            try:
                result = original(path, *args, **kwargs)
            except Exception as exc:
                details["error"] = f"{type(exc).__name__}: {exc}"
                # Let upstream apply its original fallback unchanged.
                raise
            details["samples"] = int(result[1].shape[0])
            return result
        return call

    try:
        for arm in ("left", "right"):
            planner = getattr(env.robot, f"{arm}_mplib_planner", None)
            if planner is not None:
                original = planner.TOPP
                originals.append((planner, original))
                planner.TOPP = observed_topp(arm, original)
        env.take_action(action, action_type="qpos")
    except Exception as exc:
        record["execution_error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for planner, original in originals:
            planner.TOPP = original
        record.update(
            step_after=env.take_action_cnt,
            eval_success=bool(env.eval_success),
            fail_flag_after=getattr(env, "fail_flag", None),
            task_counter_after=getattr(env, "get_obs_cnt", None),
        )
        output = Path(trace_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "execution.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
