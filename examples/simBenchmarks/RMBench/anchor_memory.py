"""RMBench-only episode anchor contract; generic rolling WM remains unchanged."""
from collections import deque

import numpy as np
from PIL import Image


def memory_contract(framework):
    anchor = framework.get("rmbench_anchor", None) or {}
    if not anchor.get("enabled", False):
        return None
    wm = framework.get("working_memory", None) or {}
    contract = {
        "version": 1,
        "history_frames": int(wm.get("history_frames", 0)),
        "history_stride": int(wm.get("history_stride", 2)),
        "history_image_size": int(wm.get("history_image_size", 112)),
        "anchor_image_size": int(anchor.get("image_size", 112)),
        "current_image_size": 224,
        "mrope_t_stride": int(wm.get("mrope_t_stride", 2)),
        "history_view_key": wm.get("history_view_key", "video.cam_high"),
    }
    if contract["history_frames"] not in (3, 4):
        raise ValueError("RMBench anchor requires 4 recent frames (3 only for explicit 7-image fallback)")
    if contract["history_view_key"] != "video.cam_high":
        raise ValueError("RMBench anchor and WM must use the head camera video.cam_high")
    for key in ("history_stride", "history_image_size", "anchor_image_size", "mrope_t_stride"):
        if contract[key] <= 0:
            raise ValueError(f"{key} must be positive")
    contract["image_roles"] = ["episode_anchor"] + [
        f"recent_history_{i}" for i in range(contract["history_frames"])
    ] + ["current_head", "current_left_wrist", "current_right_wrist"]
    return contract


def frame_ids(step, contract):
    if step < 0:
        raise ValueError("frame index must be nonnegative")
    n, stride = contract["history_frames"], contract["history_stride"]
    return [0] + [max(0, step - i * stride) for i in range(n, 0, -1)] + [step] * 3


def resize_memory(frame, size):
    return Image.fromarray(np.asarray(frame)).resize((size, size), Image.Resampling.BILINEAR)


def observation_metadata(step, contract, episode=None):
    return {"episode_id": None if episode is None else int(episode),
            "frame_ids": frame_ids(step, contract),
            "image_roles": list(contract["image_roles"]),
            "frame_id_units": "observation_index"}


class EpisodeAnchorBuffer:
    """Cache the first visible head image and raw recent heads per environment."""
    def __init__(self, contract):
        self.contract = contract
        self.history = deque(maxlen=contract["history_frames"] * contract["history_stride"] + 1)
        self.reset()

    def reset(self):
        self.anchor = None
        self.history.clear()
        self.last_step = -1

    def observe(self, frame, step):
        if step == self.last_step:
            return
        if step != self.last_step + 1:
            raise ValueError(f"RMBench memory needs every observation from episode start: {self.last_step} -> {step}")
        frame = np.asarray(frame).copy()
        if self.anchor is None:
            self.anchor = frame.copy()
        self.history.append(frame)
        self.last_step = step

    def pack(self, current_views):
        from starVLA.model.modules.vlm.working_memory import select_history_indices
        if self.anchor is None:
            raise ValueError("Episode anchor is not initialized")
        c = self.contract
        indices = select_history_indices(len(self.history), c["history_frames"], c["history_stride"])
        return [np.asarray(resize_memory(self.anchor, c["anchor_image_size"]))] + [
            np.asarray(resize_memory(self.history[i], c["history_image_size"])) for i in indices
        ] + list(current_views)
