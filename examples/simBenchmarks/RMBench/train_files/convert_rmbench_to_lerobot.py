"""RMBench — convert raw RoboTwin-style ``data/<task>/<split>/`` hdf5 episodes
into the LeRobot v2.1 layout consumed by ``starVLA.dataloader.gr00t_lerobot``.

Raw layout (HF ``TianxingChen/RMBench``, also the ``mmlab-franka`` repackaging).
Locally the raw data lives at ``/data/starVLA/RMBench`` (downloaded via
``$RMBENCH_HOME/data/_download.py`` from the upstream checkout, which redirects
output there), i.e. ``--raw-root /data/starVLA/RMBench`` covers
``data/<task>/demo_clean/`` for all 12 tasks::

    <raw_root>/<task>/<split>/        # local: data/<task>/demo_clean/
        data/episode{N}.hdf5            # joint_action/* + observation/<cam>/rgb (jpeg bits)
        video/episode{N}.mp4            # head-camera preview only -- NOT used
        instructions/episode{N}.json    # {"seen": [...], "unseen": [...]}
        language_annotation.json
        scene_info.json
        seed.txt

hdf5 internals (see RMBench ``envs/utils/pkl2hdf5.py``)::

    joint_action/left_arm      (T, 6)   joint_action/right_arm     (T, 6)
    joint_action/left_gripper  (T,)     joint_action/right_gripper (T,)
    joint_action/vector        (T, 14)  # [left_arm, left_gripper, right_arm, right_gripper]
    observation/head_camera/rgb   (T,)  jpeg bytes (null-padded fixed-length strings)
    observation/left_camera/rgb   (T,)  # left wrist
    observation/right_camera/rgb  (T,)  # right wrist

Output layout (LeRobot v2.1 + gr00t modality.json), one dataset per task,
with all of its splits merged and episodes re-indexed contiguously::

    <out_root>/<task>/
        meta/info.json
        meta/episodes.jsonl
        meta/tasks.jsonl
        meta/modality.json
        meta/embodiment.json
        data/chunk-000/episode_NNNNNN.parquet
        videos/chunk-000/observation.images.<cam>/episode_NNNNNN.mp4

State / action convention (``abs_qpos``, same as the official RMBench Mem-0
converter ``policy/Mem-0/scripts/hdf5_to_lerobot/M1_dataset_to_lerobot.py``):

    observation.state[t] (14,) = joint_action/vector[t]
    action[t]            (14,) = joint_action/vector[t + 1]   (last frame repeats itself)

The 14-dim layout matches starVLA's existing ``robotwin`` Agilex data config
(``examples/simBenchmarks/Robotwin/train_files/data_registry/data_config.py``),
so converted datasets reuse its ``robotwin`` (action chunk 16) / ``robotwin50``
(action chunk 50) data configs via the mixtures in this benchmark's own
``train_files/data_registry/data_config.py`` (e.g. ``rmbench_all``).

``meta/stats_gr00t.json`` and ``meta/steps_data_index.pkl`` are intentionally NOT
written: starVLA's dataloader computes both on first use.

Usage::

    python examples/simBenchmarks/RMBench/train_files/convert_rmbench_to_lerobot.py \
        --raw-root /data/starVLA/RMBench \
        --out-root /data/starVLA/rmbench_lerobot \
        --tasks rearrange_blocks           # optional; default = all tasks found
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# hdf5 / jpeg decode / mp4 encode. cv2 + av are already starVLA deps;
# h5py may need `pip install h5py` in the training env.
import h5py
import cv2
import av

FPS = 30  # RMBench/RoboTwin 2.0 collection rate (official Mem-0 converter uses fps=30)
CHUNK_SIZE = 1000  # LeRobot episodes per chunk

# RMBench camera name -> starVLA/robotwin video key.
CAMERA_MAP = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}

# Task-level fallback instructions (from the official Mem-0 converter) when an
# episode's instructions/*.json is missing.
TASK_INSTRUCTIONS = {
    "observe_and_pickup": "Initially, there is one target object on the shelf and five random objects on the table. Then, a screen obscures the target object. Pick up the corresponding target object from the table and lift it up.",
    "put_back_block": "There are four mats, one block, and a button on the table. One block is on one of the mats. First, put the block to the center, then press the button. Then, put the block back in its original position.",
    "rearrange_blocks": "Move the block between the two mats onto the empty mat, press the button, then move the other block (the one that started on a mat) to the space between the two mats.",
    "swap_blocks": "There are three traies on the table, and two blocks are placed in two different traies. You may move only one block at a time, and each tray can hold at most one block. Swap the positions of the two blocks. Finally press the button.",
    "swap_T": "Swap the poses of the two T-blocks, including both position and orientation.",
    # Verified against the on-disk instructions/*.json of these two tasks.
    "classify_blocks": "There are two colors of blocks and two baskets on the table. Collect blocks of the same color into the same basket.",
    "storage_blocks": "There are blocks and a basket on the table. Store all the blocks on the table into the basket.",
}

STATE_DIM = 14  # [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]


# ---------------------------------------------------------------------------
# gr00t modality.json — identical layout to the Robotwin example (14-dim Agilex)
# ---------------------------------------------------------------------------
def _modality_json(video_keys: list[str], cam_original_keys: dict[str, str]) -> dict:
    def _span(start: int, end: int, original_key: str) -> dict:
        return {"start": start, "end": end, "original_key": original_key}

    return {
        "state": {
            "left_joints": _span(0, 6, "observation.state"),
            "left_gripper": _span(6, 7, "observation.state"),
            "right_joints": _span(7, 13, "observation.state"),
            "right_gripper": _span(13, 14, "observation.state"),
        },
        "action": {
            "left_joints": _span(0, 6, "action"),
            "left_gripper": _span(6, 7, "action"),
            "right_joints": _span(7, 13, "action"),
            "right_gripper": _span(13, 14, "action"),
        },
        "video": {vk: {"original_key": cam_original_keys[vk]} for vk in video_keys},
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    }


def _build_features(video_keys: list[str], cam_original_keys: dict[str, str],
                    height: int, width: int, fps: int) -> dict:
    feats: dict = {}
    for vk in video_keys:
        feats[cam_original_keys[vk]] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": height, "video.width": width, "video.channels": 3,
                "video.fps": fps, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False, "has_audio": False,
            },
        }
    feats["observation.state"] = {"dtype": "float32", "shape": [STATE_DIM], "names": ["state"]}
    feats["action"] = {"dtype": "float32", "shape": [STATE_DIM], "names": ["actions"]}
    feats["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    feats["frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
    feats["episode_index"] = {"dtype": "int64", "shape": [1], "names": None}
    feats["index"] = {"dtype": "int64", "shape": [1], "names": None}
    feats["task_index"] = {"dtype": "int64", "shape": [1], "names": None}
    return feats


# ---------------------------------------------------------------------------
# hdf5 reading helpers
# ---------------------------------------------------------------------------
def _decode_jpeg(bits) -> np.ndarray:
    """Decode one null-padded jpeg byte string to an RGB image (H, W, 3)."""
    img = cv2.imdecode(np.frombuffer(bytes(bits), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed on observation rgb bits")
    # RMBench stores the array named 'rgb' via cv2.imencode, so decode returns
    # the same channel order it was given: RGB.
    return img


def _detect_cameras(h5: "h5py.File") -> dict[str, str]:
    """Map available RMBench camera groups to starVLA video keys."""
    available = {}
    for raw_cam, video_key in CAMERA_MAP.items():
        ds = h5.get(f"observation/{raw_cam}/rgb")
        if ds is not None:
            available[raw_cam] = video_key
    return available


def _encode_camera_video(rgb_ds, out_path: Path, fps: int, expected_frames: int) -> tuple[int, int, int]:
    """Stream-decode jpeg bits -> h264/yuv420p mp4. Returns (n_frames, height, width)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(out_path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "veryfast"}
        n_frames = 0
        height = width = 0
        for frame_idx in range(expected_frames):
            frame_rgb = _decode_jpeg(rgb_ds[frame_idx])
            if n_frames == 0:
                height, width = frame_rgb.shape[:2]
                if width % 2 or height % 2:
                    raise ValueError(f"odd frame size {width}x{height}; libx264 yuv420p needs even dims")
                stream.width = width
                stream.height = height
            frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
            n_frames += 1
        for packet in stream.encode(None):  # flush
            container.mux(packet)
        return n_frames, height, width
    finally:
        container.close()


def _write_parquet(df_path: Path, state: np.ndarray, action: np.ndarray,
                   episode_index: int, task_index: int, global_offset: int,
                   fps: int) -> int:
    n = state.shape[0]
    table = pa.table({
        "observation.state": pa.array([row.tolist() for row in state], type=pa.list_(pa.float32(), STATE_DIM)),
        "action":            pa.array([row.tolist() for row in action], type=pa.list_(pa.float32(), STATE_DIM)),
        "timestamp":         pa.array(np.arange(n, dtype=np.float32) / fps, type=pa.float32()),
        "frame_index":       pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
        "episode_index":     pa.array(np.full(n, episode_index, dtype=np.int64), type=pa.int64()),
        "index":             pa.array(np.arange(global_offset, global_offset + n, dtype=np.int64), type=pa.int64()),
        "task_index":        pa.array(np.full(n, task_index, dtype=np.int64), type=pa.int64()),
    })
    df_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, df_path)
    return n


def _load_instruction(split_dir: Path, ep_num: int, task: str) -> str:
    instr_path = split_dir / "instructions" / f"episode{ep_num}.json"
    if instr_path.is_file():
        try:
            data = json.loads(instr_path.read_text())
            for key in ("seen", "unseen"):
                vals = data.get(key)
                if vals:
                    return str(vals[0])
        except (json.JSONDecodeError, IndexError, TypeError):
            pass
    return TASK_INSTRUCTIONS.get(task, task.replace("_", " "))


# ---------------------------------------------------------------------------
# Per-task conversion
# ---------------------------------------------------------------------------
def convert_task(raw_root: Path, task: str, out_root: Path, splits: list[str] | None,
                 max_episodes: int | None, overwrite: bool, fps: int) -> dict:
    raw_task = raw_root / task
    if not raw_task.is_dir():
        raise FileNotFoundError(f"Raw task dir not found: {raw_task}")

    # Discover splits: subdirs that contain a data/ folder with hdf5 episodes.
    split_dirs = []
    for cand in sorted(raw_task.iterdir()):
        if not cand.is_dir():
            continue
        if splits is not None and cand.name not in splits:
            continue
        if (cand / "data").is_dir() and list((cand / "data").glob("episode*.hdf5")):
            split_dirs.append(cand)
    if not split_dirs:
        raise RuntimeError(f"No splits with episodes under {raw_task} (splits={splits})")

    out_task = out_root / task
    if out_task.exists():
        if overwrite:
            print(f"[convert] removing existing output {out_task}")
            shutil.rmtree(out_task)
        else:
            raise FileExistsError(f"Output {out_task} exists (pass --overwrite).")
    (out_task / "meta").mkdir(parents=True, exist_ok=True)

    episodes_meta: list[dict] = []
    tasks_meta: dict[str, int] = {}  # instruction -> task_index
    total_frames = 0
    total_videos = 0
    new_ep_idx = 0
    probed_height = probed_width = 0
    video_keys: list[str] = []
    cam_original_keys: dict[str, str] = {}

    for split_dir in split_dirs:
        hdf5_files = sorted(
            (split_dir / "data").glob("episode*.hdf5"),
            key=lambda p: int(p.stem.replace("episode", "")),
        )
        for hdf5_path in hdf5_files:
            if max_episodes is not None and new_ep_idx >= max_episodes:
                break
            ep_num = int(hdf5_path.stem.replace("episode", ""))
            try:
                with h5py.File(hdf5_path, "r") as h5:
                    if "joint_action/vector" not in h5:
                        raise KeyError("missing joint_action/vector")
                    qpos = np.asarray(h5["joint_action/vector"], dtype=np.float32)
                    if qpos.ndim != 2 or qpos.shape[1] != STATE_DIM:
                        raise ValueError(f"joint_action/vector shape {qpos.shape}, expected (T, {STATE_DIM})")
                    cams = _detect_cameras(h5)
                    if "head_camera" not in cams:
                        raise KeyError("missing observation/head_camera/rgb")
                    T = qpos.shape[0]

                    # Probe / validate camera layout from the first episode.
                    if new_ep_idx == 0:
                        video_keys = [cams[c] for c in CAMERA_MAP if c in cams]
                        cam_original_keys = {vk: f"observation.images.{vk}" for vk in video_keys}
                        missing = [c for c in CAMERA_MAP if c not in cams]
                        if missing:
                            print(f"[convert][WARN] {task}: cameras {missing} not in hdf5; "
                                  f"converting with {video_keys} only")
                    elif [cams[c] for c in CAMERA_MAP if c in cams] != video_keys:
                        raise ValueError("camera set changes across episodes; not supported")

                    chunk = new_ep_idx // CHUNK_SIZE
                    # Encode one video per camera (streaming, low memory).
                    for raw_cam, video_key in cams.items():
                        mp4_out = (out_task / f"videos/chunk-{chunk:03d}/"
                                   f"observation.images.{video_key}/episode_{new_ep_idx:06d}.mp4")
                        n_frames, h, w = _encode_camera_video(h5[f"observation/{raw_cam}/rgb"], mp4_out, fps, T)
                        if n_frames != T:
                            raise ValueError(f"{raw_cam}: {n_frames} frames != {T} joint steps")
                        if new_ep_idx == 0:
                            probed_height, probed_width = h, w
                        elif (h, w) != (probed_height, probed_width):
                            raise ValueError(f"{raw_cam}: resolution {w}x{h} differs from first episode "
                                             f"{probed_width}x{probed_height}; not supported")
                        total_videos += 1

                # state[t] = qpos[t]; action[t] = qpos[t+1] (last frame repeats).
                state = qpos
                action = np.concatenate([qpos[1:], qpos[-1:]], axis=0)

                instruction = _load_instruction(split_dir, ep_num, task)
                if instruction not in tasks_meta:
                    tasks_meta[instruction] = len(tasks_meta)
                task_index = tasks_meta[instruction]

                pq_path = out_task / f"data/chunk-{chunk:03d}/episode_{new_ep_idx:06d}.parquet"
                _write_parquet(pq_path, state, action, new_ep_idx, task_index, total_frames, fps)

                episodes_meta.append({"episode_index": new_ep_idx, "tasks": [instruction], "length": T})
                total_frames += T
                new_ep_idx += 1
                if new_ep_idx % 25 == 0:
                    print(f"[convert][{task}] {new_ep_idx} episodes ({total_frames} frames)")
            except Exception as exc:  # skip broken episode, keep going
                print(f"[convert][skip] {task}/{split_dir.name}/{hdf5_path.name}: {exc}")

    if new_ep_idx == 0:
        raise RuntimeError(f"No episodes converted for task {task}")

    # ---- meta files ----
    (out_task / "meta/episodes.jsonl").write_text(
        "\n".join(json.dumps(e) for e in episodes_meta) + "\n")
    (out_task / "meta/tasks.jsonl").write_text(
        "\n".join(json.dumps({"task_index": idx, "task": instr})
                  for instr, idx in sorted(tasks_meta.items(), key=lambda kv: kv[1])) + "\n")
    (out_task / "meta/modality.json").write_text(
        json.dumps(_modality_json(video_keys, cam_original_keys), indent=4))
    (out_task / "meta/embodiment.json").write_text(json.dumps({
        "robot_name": "aloha_agilex",
        "robot_type": "aloha_agilex_dual_arm",
        "record_frequency": fps,
        "embodiment_tag": "robotwin",
    }, indent=2))

    info = {
        "codebase_version": "v2.1",
        "robot_type": "aloha_agilex",
        "total_episodes": new_ep_idx,
        "total_frames": total_frames,
        "total_tasks": len(tasks_meta),
        "total_videos": total_videos,
        "total_chunks": (new_ep_idx + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{new_ep_idx}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": _build_features(video_keys, cam_original_keys, probed_height, probed_width, fps),
    }
    (out_task / "meta/info.json").write_text(json.dumps(info, indent=4))

    summary = {"task": task, "episodes": new_ep_idx, "frames": total_frames,
               "cameras": video_keys, "resolution": [probed_height, probed_width],
               "instructions": len(tasks_meta)}
    print(f"[convert][{task}] done: {summary}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw-root", required=True, type=Path,
                   help="RMBench raw root containing data/<task>/<split>/ (or the <task> level directly)")
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Task names to convert (default: all task dirs found)")
    p.add_argument("--splits", nargs="*", default=None,
                   help="Split names to include (default: all splits with episodes, e.g. demo_clean + supplements)")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Cap converted episodes per task (useful for smoke tests)")
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    raw_root = args.raw_root
    # Accept either the repo-style root (contains data/<task>/...) or the data/ dir itself.
    data_root = raw_root / "data" if (raw_root / "data").is_dir() else raw_root

    if args.tasks is not None:
        tasks = args.tasks
    else:
        tasks = sorted(d.name for d in data_root.iterdir() if d.is_dir())
    if not tasks:
        sys.exit(f"[convert] no tasks found under {data_root}")

    print(f"[convert] tasks: {tasks}")
    summaries = []
    for task in tasks:
        summaries.append(convert_task(data_root, task, args.out_root, args.splits,
                                      args.max_episodes, args.overwrite, args.fps))
    total_eps = sum(s["episodes"] for s in summaries)
    total_frames = sum(s["frames"] for s in summaries)
    print(f"\n[convert] ALL DONE: {len(summaries)} tasks, {total_eps} episodes, {total_frames} frames")
    print(f"[convert] output -> {args.out_root}")
    print("[convert] next: register these tasks in "
          "examples/simBenchmarks/RMBench/train_files/data_registry/data_config.py")


if __name__ == "__main__":
    main()
