# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""RMBench env-side adapter (thin WebSocket client) for starVLA checkpoints.

Loaded by `$RMBENCH_HOME/script/eval_policy.py` as `policy_name` module
(`sys.path.append("./policy")` — eval_rmbench.sh symlinks this file to
`$RMBENCH_HOME/policy/starvla_rmbench.py`). The heavy model runs in the
starVLA policy server (`run_policy_server.sh`); this module only:

  1. converts RMBench observations to the training-time contract, and
  2. converts model action chunks back to env layout.

Layout contract (verified against dataset_statistics.json + raw parquets):
  - env / raw hdf5 / parquet 14-dim vector:
      [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
  - model (AgilexData50Config state_keys/action_keys order):
      [left_arm(6), right_arm(6), left_gripper(1), right_gripper(1)]

State normalization (training transform, replicated here with plain numpy so
this file stays importable in the RMBench conda env):
  - joints   (model dims 0:12): min_max -> [-1, 1]; dims with min == max -> 0
  - grippers (model dims 12:14): binary, x > 0.49 -> 1.0 else 0.0
The server un-normalizes predicted actions with the same training transforms.

Images: [head_camera, left_camera, right_camera] resized to 224x224, matching
training (`obs_image_size: [224, 224]`; cam order = modality.json video keys).
Optional working memory: when `history_frames > 0`, the last
`history_frames * history_stride + 1` head-camera frames are buffered and
`history_frames` downsampled 112x112 history views are prepended (oldest
first), mirroring the dataloader's `pack_step_images` layout. Keep this OFF
(0) for checkpoints trained without `framework.working_memory`.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

# env layout <-> model layout permutation indices (see module docstring).
ENV_TO_MODEL = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]
MODEL_TO_ENV = [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]

_N_JOINT_DIMS = 12  # model layout: first 12 dims are arm joints, last 2 grippers
_BINARY_THRESHOLD = 0.49  # AgilexData50Config StateActionTransform binary_threshold


def _find_stats_path(policy_ckpt_path: str, stats_path: Optional[str]) -> Path:
    """Locate dataset_statistics.json next to the checkpoint run dir."""
    if stats_path:
        p = Path(stats_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"stats_path not found: {p}")
    ckpt = Path(policy_ckpt_path)
    candidates = []
    if ckpt.is_file():
        # .../<RUN>/checkpoints/steps_X_pytorch_model.pt -> <RUN>/dataset_statistics.json
        candidates.append(ckpt.parents[1] / "dataset_statistics.json")
    candidates += [
        ckpt / "dataset_statistics.json",
        ckpt.parent / "dataset_statistics.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"dataset_statistics.json not found near {policy_ckpt_path}; "
        "pass stats_path explicitly in deploy_policy.yml / --overrides."
    )


def _load_state_stats(policy_ckpt_path: str, stats_path: Optional[str], unnorm_key: Optional[str]):
    """Return (min, max) float32 arrays (14,) in MODEL layout for state normalization."""
    path = _find_stats_path(policy_ckpt_path, stats_path)
    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    if unnorm_key is None:
        if len(stats) != 1:
            raise ValueError(
                f"dataset_statistics.json has multiple keys {list(stats.keys())}; "
                "set unnorm_key to pick one."
            )
        unnorm_key = next(iter(stats))
    state = stats[unnorm_key]["state"]
    s_min = np.asarray(state["min"], dtype=np.float32).reshape(-1)
    s_max = np.asarray(state["max"], dtype=np.float32).reshape(-1)
    if s_min.shape[0] != 14:
        raise ValueError(f"expected 14-dim state stats, got {s_min.shape} from {path}")
    print(f"[model2rmbench] loaded state stats from {path} (key={unnorm_key})")
    return s_min, s_max


class ModelClient:
    """Chunk-cached WebSocket client matching the RMBench training contract."""

    def __init__(
        self,
        policy_ckpt_path: str,
        host: str = "127.0.0.1",
        port: int = 5694,
        unnorm_key: Optional[str] = None,
        stats_path: Optional[str] = None,
        image_size: Sequence[int] = (224, 224),
        history_frames: Optional[int] = None,
        history_stride: Optional[int] = None,
        history_image_size: Optional[Sequence[int]] = None,
        trace_dir: Optional[str] = None,
    ) -> None:
        self.client = WebsocketClientPolicy(host, port)
        meta = self.client.get_server_metadata()
        self.action_chunk_size = int(meta["action_chunk_size"])
        print(
            f"*** model2rmbench: ckpt={policy_ckpt_path} server={host}:{port} "
            f"action_chunk_size={self.action_chunk_size} unnorm_key={unnorm_key} "
            f"history_frames={history_frames} server_meta={meta} ***"
        )

        self.unnorm_key = unnorm_key
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.trace_episode_dir: Optional[Path] = None
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.state_min, self.state_max = _load_state_stats(policy_ckpt_path, stats_path, unnorm_key)

        # Working memory (keep 0 for checkpoints trained without WM).
        self.anchor_contract = meta.get("rmbench_anchor")
        wm = meta.get("training_working_memory", {}) or {}
        expected = self.anchor_contract or wm
        default_size = int(expected.get("history_image_size", 112))
        self.history_frames = int(expected.get("history_frames", 0) if history_frames is None else history_frames)
        self.history_stride = int(expected.get("history_stride", 2) if history_stride is None else history_stride)
        self.history_image_size = tuple(history_image_size or (default_size, default_size))
        if self.anchor_contract:
            if (self.history_frames != expected["history_frames"] or self.history_stride != expected["history_stride"]
                    or self.history_image_size != (default_size, default_size) or self.image_size != (224, 224)):
                raise ValueError("RMBench memory overrides conflict with the checkpoint contract")
            from examples.simBenchmarks.RMBench.anchor_memory import EpisodeAnchorBuffer
            self.anchor_buffer = EpisodeAnchorBuffer(self.anchor_contract)
        else:
            self.anchor_buffer = None
        wm_maxlen = self.history_frames * self.history_stride + 1 if self.history_frames > 0 else 1
        self.image_history = deque(maxlen=wm_maxlen)

        self.task_description: Optional[str] = None
        self.raw_actions: Optional[np.ndarray] = None  # cached chunk (T, 14), env layout

    # ------------------------------------------------------------------
    # episode lifecycle
    # ------------------------------------------------------------------
    def reset(self, task_description: str = "") -> None:
        self.task_description = task_description
        self.image_history.clear()
        self.raw_actions = None
        if self.anchor_buffer is not None:
            self.anchor_buffer.reset()

    # ------------------------------------------------------------------
    # observation conversion
    # ------------------------------------------------------------------
    def _resize(self, image: np.ndarray, hw, resample=Image.BILINEAR) -> np.ndarray:
        arr = np.asarray(image)
        if arr.shape[:2] != hw:
            arr = np.asarray(Image.fromarray(arr).resize((hw[1], hw[0]), resample))
        return arr

    def _normalize_state(self, state_env: np.ndarray) -> np.ndarray:
        """env layout raw qpos (14,) -> normalized model layout (1, 14)."""
        s = np.asarray(state_env, dtype=np.float32).reshape(-1)[ENV_TO_MODEL]
        out = np.zeros_like(s)
        joints = s[:_N_JOINT_DIMS]
        lo, hi = self.state_min[:_N_JOINT_DIMS], self.state_max[:_N_JOINT_DIMS]
        mask = lo != hi
        out[:_N_JOINT_DIMS][mask] = 2.0 * (joints[mask] - lo[mask]) / (hi[mask] - lo[mask]) - 1.0
        out[_N_JOINT_DIMS:] = (s[_N_JOINT_DIMS:] > _BINARY_THRESHOLD).astype(np.float32)
        # LeRobotSingleDataset._pack_sample casts state to float16 before
        # QwenPI_v3 discretizes it. Preserve that rounding at token boundaries.
        return out.astype(np.float16).reshape(1, -1)

    def _prepend_history(self, images: list) -> list:
        """Prepend [hist_0..hist_{F-1}] (oldest->newest, 112x112) to current views."""
        from starVLA.model.modules.vlm.working_memory import select_history_indices

        n = len(self.image_history)
        hist = [
            self._resize(np.asarray(self.image_history[idx]), self.history_image_size)
            for idx in select_history_indices(n, self.history_frames, self.history_stride)
        ]
        return hist + list(images)

    # ------------------------------------------------------------------
    # main step
    # ------------------------------------------------------------------
    def step(self, example: dict, step: int = 0) -> np.ndarray:
        """One env step; returns a 14-dim action in env layout (qpos)."""
        instruction = example.get("lang", None)
        if instruction != self.task_description:
            if self.anchor_buffer is None:
                self.reset(instruction)
            else:
                # Language changes within an episode must not erase its anchor.
                self.task_description = instruction
                self.raw_actions = None

        # pack_step_images uses PIL's RGB default (bicubic) for current views.
        images = [self._resize(img, self.image_size, Image.BICUBIC) for img in example["image"]]
        if self.anchor_buffer is not None:
            self.anchor_buffer.observe(example["image"][0], step)
            images = self.anchor_buffer.pack(images)
        elif self.history_frames > 0:
            self.image_history.append(images[0])  # head camera is the history view
            images = self._prepend_history(images)

        if step % self.action_chunk_size == 0 or self.raw_actions is None:
            # === TRAIN/TEST CONSISTENCY: keep the observation below aligned with training ===
            #   - state       : normalized (min_max joints / binary gripper), MODEL layout
            #   - image size  : 224x224 current views (+ 112x112 history when WM on)
            #   - image order : [cam_high, cam_left_wrist, cam_right_wrist]
            #   - unnorm_key  : must match the training dataset stats
            # ==============================================================================
            vla_input = {
                "examples": [
                    {
                        "image": images,
                        "lang": instruction,
                        "state": self._normalize_state(example["state"]),
                    }
                ],
                "unnorm_key": self.unnorm_key,
                "do_sample": False,
            }
            if self.anchor_contract:
                from examples.simBenchmarks.RMBench.anchor_memory import observation_metadata
                vla_input["examples"][0]["rmbench_memory"] = observation_metadata(step, self.anchor_contract, example.get("episode_id"))
            request_start = time.perf_counter()
            response = self.client.predict_action(vla_input)
            self.last_inference_ms = (time.perf_counter() - request_start) * 1000
            if response.get("status") == "error" or response.get("ok") is False:
                raise RuntimeError(f"Policy inference failed: {response.get('error')}")
            actions = np.asarray(response["data"]["actions"])[0]  # (T, 14) model layout
            self.raw_actions = actions[:, MODEL_TO_ENV]  # -> env layout
            self.last_memory_metadata = vla_input["examples"][0].get("rmbench_memory")
            self.last_visual_layout = response["data"].get("rmbench_visual_layout")
            if self.anchor_contract:
                print(f"[RMBENCH_ANCHOR_CLIENT] step={step} inference_roundtrip_ms={self.last_inference_ms:.2f}", flush=True)
            if self.trace_episode_dir is not None:
                self.trace_episode_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    self.trace_episode_dir / f"inference_{step:06d}.npz",
                    state_env=np.asarray(example["state"]),
                    state_input=vla_input["examples"][0]["state"],
                    actions_model=actions,
                    actions_env=self.raw_actions,
                    instruction=np.asarray(str(instruction)),
                    memory_metadata=np.asarray(json.dumps(self.last_memory_metadata)),
                    visual_layout=np.asarray(json.dumps(self.last_visual_layout)),
                    inference_roundtrip_ms=self.last_inference_ms,
                    **{f"model_image_{i}": img for i, img in enumerate(images)},
                    **{f"raw_image_{i}": img for i, img in enumerate(example["image"])},
                )

        return self.raw_actions[step % self.action_chunk_size]


# ---------------------------------------------------------------------------
# RMBench policy module contract (script/eval_policy.py)
# ---------------------------------------------------------------------------
def get_model(usr_args: dict) -> ModelClient:
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    if not policy_ckpt_path:
        raise ValueError("policy_ckpt_path must be set (deploy_policy.yml or --overrides)")
    return ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=usr_args.get("host", "127.0.0.1"),
        port=int(usr_args.get("port", 5694)),
        unnorm_key=usr_args.get("unnorm_key", None),
        stats_path=usr_args.get("stats_path", None),
        image_size=usr_args.get("image_size", (224, 224)),
        history_frames=usr_args.get("history_frames"),
        history_stride=usr_args.get("history_stride"),
        history_image_size=usr_args.get("history_image_size"),
        trace_dir=usr_args.get("trace_dir") or os.environ.get("RMBENCH_TRACE_DIR"),
    )


def reset_model(model: ModelClient) -> None:
    model.reset()


def eval(TASK_ENV, model: ModelClient, observation: dict) -> None:
    if model.trace_dir is not None:
        model.trace_episode_dir = (
            model.trace_dir / TASK_ENV.task_name / f"episode_{TASK_ENV.ep_num:06d}"
        )
    instruction = str(TASK_ENV.get_instruction())
    obs = observation["observation"]
    images = [
        obs["head_camera"]["rgb"],
        obs["left_camera"]["rgb"],
        obs["right_camera"]["rgb"],
    ]
    example = {
        "lang": instruction,
        "episode_id": getattr(TASK_ENV, "ep_num", None),
        "image": images,
        "state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
    }
    action = model.step(example, step=TASK_ENV.take_action_cnt)
    if model.trace_episode_dir is None:
        TASK_ENV.take_action(action, action_type="qpos")
    else:
        from examples.simBenchmarks.RMBench.eval_files.execution_diagnostics import take_action_with_diagnostics

        take_action_with_diagnostics(TASK_ENV, action, model.trace_episode_dir)
