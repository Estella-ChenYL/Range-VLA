"""Working Memory v1: history frames as plain multi-image input.

Pure helpers shared by the dataloader, the Qwen3-VL interface and the LIBERO
eval client. This module must import ONLY torch/numpy/PIL so it stays testable
without the training image's heavy dependency chain (pytorch3d etc.).

Layout contract (order is load-bearing):
    example["image"] = [hist_0..hist_{F-1} (small), current views... (full size)]
The first F images are history (oldest -> newest); the rest are the current
observation's views. M-RoPE T channel: history image i gets T = i * t_stride,
all current views share T = F * t_stride.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

CURRENT_IMAGE_SIZE = 224  # legacy dataloader behaviour, do not change


def history_delta_indices(history_frames: int, history_stride: int) -> list[int]:
    """Dataloader delta indices for the history view, e.g. (4, 2) -> [-8,-6,-4,-2,0]."""
    return [-(i * history_stride) for i in range(history_frames, 0, -1)] + [0]


def select_history_indices(n_available: int, history_frames: int, history_stride: int) -> list[int]:
    """Eval-side: pick `history_frames` entries (oldest->newest) from a deque holding
    the most recent `n_available` frames (index n_available-1 == current).

    Mirrors the dataloader's np.maximum(step_indices, 0) clamp: before enough
    history exists, the earliest buffered frame is duplicated (at episode start
    that is the current observation itself). The upper clamp ``min(..., n_available-1)``
    is an intentional defensive guard beyond the dataloader's max(·,0) clamp.
    """
    return [
        min(max(n_available - 1 - j * history_stride, 0), n_available - 1)
        for j in range(history_frames, 0, -1)
    ]


def pack_step_images(frames_by_key: dict, video_keys: list[str], wm_cfg: dict | None) -> list:
    """Build the ordered example["image"] list from per-key frame sequences.

    frames_by_key[key]: sequence of (H, W, C) uint8 frames, one per delta index
    (oldest first, current last). With wm_cfg off (None or history_frames=0)
    this reproduces the legacy behaviour exactly: frame [0] of each key at 224.
    """
    wm = wm_cfg or {}
    n_hist = int(wm.get("history_frames", 0) or 0)
    hist_key = wm.get("history_view_key", "video.primary_image")
    hist_size = int(wm.get("history_image_size", 112))

    if n_hist > 0:
        assert video_keys.index(hist_key) == 0, (
            f"history view {hist_key} must be the first video key, got order {video_keys}"
        )

    step_images = []
    for key in video_keys:
        frames = frames_by_key[key]
        if n_hist > 0 and key == hist_key:
            assert len(frames) == n_hist + 1, (
                f"{key}: expected {n_hist + 1} frames (history+current), got {len(frames)}"
            )
            for fr in frames[:-1]:
                step_images.append(Image.fromarray(fr).resize((hist_size, hist_size), Image.BILINEAR))
            step_images.append(Image.fromarray(frames[-1]).resize((CURRENT_IMAGE_SIZE, CURRENT_IMAGE_SIZE)))
        else:
            # wm on: non-history views share the history deltas, so the current
            # frame is the LAST entry; wm off: single frame at index 0 (legacy).
            frame = frames[-1] if n_hist > 0 else frames[0]
            step_images.append(Image.fromarray(frame).resize((CURRENT_IMAGE_SIZE, CURRENT_IMAGE_SIZE)))
    return step_images


def apply_history_mrope(
    position_ids: torch.Tensor,       # (3, B, L) output of model.get_rope_index
    mm_token_type_ids: torch.Tensor,  # (B, L), 0=text 1=image 2=video
    attention_mask: torch.Tensor,     # (B, L)
    history_frames: int,
    t_stride: int,
) -> torch.Tensor:
    """Rewrite ONLY the T channel so the leading `history_frames` images sit at
    T = 0, s, ..., (F-1)*s and every later image shares T = F*s.

    Image spans are contiguous runs of type==1 in the UNMASKED region, in order.
    In Qwen3-VL tokenization every image is wrapped in type-0
    <|vision_start|>/<|vision_end|> tokens, so each image is its own span and
    span order matches image order (the same order get_rope_index consumed
    image_grid_thw). H/W channels and text positions are left untouched.
    """
    position_ids = position_ids.clone()
    B, L = mm_token_type_ids.shape
    for b in range(B):
        valid = attention_mask[b].bool()
        valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        types = mm_token_type_ids[b][valid].tolist()
        spans, start = [], None
        for i, t in enumerate(types):
            if t == 1 and start is None:
                start = i
            elif t != 1 and start is not None:
                spans.append((start, i))
                start = None
        if start is not None:
            spans.append((start, len(types)))
        for img_idx, (s0, s1) in enumerate(spans):
            t_val = min(img_idx, history_frames) * t_stride
            position_ids[0, b, valid_idx[s0:s1]] = t_val
    return position_ids
