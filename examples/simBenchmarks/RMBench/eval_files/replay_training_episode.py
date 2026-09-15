"""Compare a policy server with a converted training demonstration.

Run from the repo root with ``python -m
examples.simBenchmarks.RMBench.eval_files.replay_training_episode --help``.
This measures open-loop action error, not simulator success rate.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .model2rmbench_interface import ModelClient


def action_metrics(predicted, expected):
    """Both inputs use the environment's [left arm, grip, right arm, grip] order."""
    predicted, expected = np.asarray(predicted), np.asarray(expected)
    if predicted.shape != expected.shape or predicted.ndim != 2 or predicted.shape[1] != 14:
        raise ValueError(f"Expected matching (T, 14) chunks, got {predicted.shape}, {expected.shape}")
    if not np.isfinite(predicted).all() or not np.isfinite(expected).all():
        raise ValueError("Non-finite predicted or demonstration actions")
    arms = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    error = np.abs(predicted - expected)
    return {
        "arm_mae_rad": float(error[:, arms].mean()),
        "arm_max_error_rad": float(error[:, arms].max()),
        "first_action_arm_mae_rad": float(error[0, arms].mean()),
        "per_joint_mae_rad": error[:, arms].mean(axis=0).tolist(),
        "gripper_accuracy": float(((predicted[:, [6, 13]] > 0.49) ==
                                   (expected[:, [6, 13]] > 0.49)).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Converted single-task LeRobot directory")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint served by the running policy server")
    parser.add_argument("--stats-path", help="Local checkpoint dataset_statistics.json, if checkpoint is remote")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5694)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 35, 50, 70, 100])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Keep --help and metric tests independent of the dataset decoding stack.
    import cv2
    import pyarrow.parquet as pq
    from PIL import Image

    info = json.loads((args.dataset / "meta/info.json").read_text())
    chunk = args.episode // info["chunks_size"]
    data_path = info["data_path"].format(episode_chunk=chunk, episode_index=args.episode)
    data = pq.read_table(args.dataset / data_path).to_pydict()
    tasks = {row["task_index"]: row["task"] for row in
             map(json.loads, (args.dataset / "meta/tasks.jsonl").read_text().splitlines())}
    keys = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    if any(step < 0 or step >= len(data["action"]) for step in args.steps):
        parser.error(f"--steps must be in [0, {len(data['action']) - 1}]")

    model = ModelClient(args.checkpoint, host=args.host, port=args.port, stats_path=args.stats_path)
    frame_cache = {}

    def read_frame(key, step):
        if (key, step) not in frame_cache:
            video = info["video_path"].format(episode_chunk=chunk, episode_index=args.episode,
                                             video_key=f"observation.images.{key}")
            cap = cv2.VideoCapture(str(args.dataset / video))
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, step)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode {video} frame {step}")
                frame_cache[key, step] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            finally:
                cap.release()
        return frame_cache[key, step]

    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for step in args.steps:
        images = [read_frame(key, step) for key in keys]
        instruction = tasks[data["task_index"][step]]
        state = np.asarray(data["observation.state"][step], dtype=np.float32)
        model.reset(instruction)
        # Reconstruct memory from this SAME episode, not from the query frame.
        if model.anchor_buffer is not None:
            for prior in range(step):
                model.anchor_buffer.observe(read_frame(keys[0], prior), prior)
        elif model.history_frames > 0:
            for prior in range(max(0, step - model.history_frames * model.history_stride), step):
                model.image_history.append(model._resize(read_frame(keys[0], prior), model.image_size, Image.BICUBIC))
        model.step({"image": images, "lang": instruction, "state": state, "episode_id": args.episode}, step=step)
        predicted = model.raw_actions.copy()
        indices = np.minimum(np.arange(step, step + len(predicted)), len(data["action"]) - 1)
        expected = np.asarray(data["action"], dtype=np.float32)[indices]
        row = {"step": step, "instruction": instruction, **action_metrics(predicted, expected)}
        results.append(row)
        np.savez_compressed(args.output / f"step_{step:04d}.npz", predicted=predicted,
                            expected=expected, state=state,
                            memory_metadata=np.asarray(json.dumps(model.last_memory_metadata)),
                            visual_layout=np.asarray(json.dumps(model.last_visual_layout)))
        for key, frame in zip(keys, images):
            Image.fromarray(frame).resize((224, 224)).save(args.output / f"step_{step:04d}_{key}.png")
        print(json.dumps(row), flush=True)
    (args.output / "metrics.json").write_text(json.dumps({
        "checkpoint": args.checkpoint, "dataset": str(args.dataset),
        "episode": args.episode, "results": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
