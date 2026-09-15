"""RMBench anchor VLM adapter. No changes to generic rolling WM or DiT."""
import copy
import inspect
import json
import logging
import time

import torch
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface, IMAGE_TOKEN_INDEX
from starVLA.model.modules.vlm.working_memory import apply_history_mrope
from .anchor_memory import memory_contract


class RMBenchImageProcessor:
    """Keep current-view preprocessing intact; permit compact memory images.

    Qwen's patch/merge alignment still applies (112 -> 128 for patch 16,
    merge 2). Group processing preserves the original flat image order.
    """
    def __init__(self, processor, contract):
        self.original = processor
        self.memory = copy.deepcopy(processor)
        self.contract = contract
        minimum = (processor.patch_size * processor.merge_size) ** 2
        self.memory.size = dict(processor.size, shortest_edge=minimum)
        if hasattr(self.memory, "min_pixels"):
            self.memory.min_pixels = minimum

    def __getattr__(self, key):
        return getattr(self.original, key)

    def __call__(self, images, **kwargs):
        flat = [img for group in images for img in group] if isinstance(images[0], (list, tuple)) else list(images)
        count = len(self.contract["image_roles"])
        if not flat or len(flat) % count:
            raise ValueError(f"RMBench anchor expects {count} images per sample")
        prefix = self.contract["history_frames"] + 1
        outputs = {}
        for memory_group, processor in ((True, self.memory), (False, self.original)):
            indices = [i for i in range(len(flat)) if (i % count < prefix) == memory_group]
            result = processor(images=[flat[i] for i in indices], **dict(kwargs, return_tensors="pt"))
            offset = 0
            for i, grid in zip(indices, result["image_grid_thw"]):
                n = int(grid.prod())
                outputs[i] = (result["pixel_values"][offset:offset + n], grid)
                offset += n
            if offset != len(result["pixel_values"]):
                raise ValueError("Unexpected Qwen pixel/grid layout")
        return BatchFeature({
            "pixel_values": torch.cat([outputs[i][0] for i in range(len(flat))]),
            "image_grid_thw": torch.stack([outputs[i][1] for i in range(len(flat))]),
        })


def visual_spans(types, attention_mask, roles):
    """Absolute token spans [start,end), separated by image role, including padding offset."""
    records = []
    for row, mask in zip(types, attention_mask):
        indices = torch.nonzero((row == 1) & mask.bool(), as_tuple=False).flatten().tolist()
        spans = []
        for idx in indices:
            if not spans or spans[-1][1] != idx:
                spans.append([idx, idx + 1])
            else:
                spans[-1][1] += 1
        if len(spans) != len(roles):
            raise ValueError(f"RMBench expected {len(roles)} image spans, found {len(spans)}")
        records.append([{"role": role, "start": a, "end": b, "tokens": b-a}
                        for role, (a, b) in zip(roles, spans)])
    return records


class RMBenchAnchorQwenVL(_QWen3_VL_Interface):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.anchor_contract = memory_contract(config.framework)
        self.processor.image_processor = RMBenchImageProcessor(self.processor.image_processor, self.anchor_contract)
        self._profile_calls = 0
        self._profile_interval = int(config.framework.rmbench_anchor.get("profile_interval", 100))

    def _maybe_apply_history_mrope(self, batch_inputs, n_images_per_sample):
        c = self.anchor_contract
        roles = c["image_roles"]
        batch_size = batch_inputs["input_ids"].shape[0]
        if n_images_per_sample != len(roles) or len(batch_inputs["image_grid_thw"]) != batch_size * len(roles):
            raise ValueError(f"RMBench anchor requires layout {roles}")
        types = batch_inputs.get("mm_token_type_ids")
        if types is None:
            types = (batch_inputs["input_ids"] == IMAGE_TOKEN_INDEX).long()
        spans = visual_spans(types, batch_inputs["attention_mask"], roles)
        rope_fn = self.model.model.get_rope_index
        options = {key: batch_inputs.get(key) for key in ("image_grid_thw", "video_grid_thw", "attention_mask")}
        if "mm_token_type_ids" in inspect.signature(rope_fn).parameters:
            options["mm_token_type_ids"] = types
        positions, _ = rope_fn(batch_inputs["input_ids"], **options)
        # Only the positional helper sees five preceding images. The WM config
        # and WM role slice remain FOUR; anchor is never a recent incident frame.
        batch_inputs["position_ids"] = apply_history_mrope(
            positions, types, batch_inputs["attention_mask"],
            history_frames=c["history_frames"] + 1, t_stride=c["mrope_t_stride"],
        )
        grids = batch_inputs["image_grid_thw"].reshape(batch_size, len(roles), 3).tolist()
        patch_size = self.processor.image_processor.patch_size
        self.last_visual_layout = [
            [dict(span, grid_thw=grid, processed_hw=[grid[1]*patch_size, grid[2]*patch_size])
             for span, grid in zip(sample, sample_grids)]
            for sample, sample_grids in zip(spans, grids)
        ]
        if not hasattr(self, "_layout_logged"):
            self._layout_logged = True
            logging.info("[RMBENCH_ANCHOR_LAYOUT] %s", json.dumps(self.last_visual_layout[0]))
        return batch_inputs

    def forward(self, **kwargs):
        self._profile_calls += 1
        profile = self._profile_calls == 1 or (self._profile_interval > 0 and self._profile_calls % self._profile_interval == 0)
        if not profile:
            return super().forward(**kwargs)
        cuda = self.model.device.type == "cuda"
        if cuda:
            torch.cuda.synchronize(self.model.device)
        start = time.perf_counter()
        result = super().forward(**kwargs)
        if cuda:
            torch.cuda.synchronize(self.model.device)
        logging.info("[RMBENCH_ANCHOR_PROFILE] %s", json.dumps({
            "vlm_forward_ms": (time.perf_counter() - start) * 1000,
            "batch_size": len(kwargs["input_ids"]),
            "visual_tokens_per_sample": [sum(s["tokens"] for s in spans) for spans in self.last_visual_layout],
            "anchor_tokens_per_sample": [spans[0]["tokens"] for spans in self.last_visual_layout],
            "process_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.model.device) if cuda else None,
            "process_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.model.device) if cuda else None,
        }))
        return result
