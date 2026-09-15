# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

from typing import Optional

import inspect
import torch
from starVLA.model.modules.vlm.working_memory import apply_history_mrope
from starVLA.model.tools import has_flash_attn  # unified flash-attn detection (GPU / NPU)
from starVLA.training.trainer_utils import initialize_overwatch
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 151669  # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = (
    153716  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
)


import torch.nn as nn


class _QWen3_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3-VL (Qwen3VLForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")
        attn_implementation = "sdpa"
        # Fallback to sdpa if flash_attention_2 is requested but flash_attn is not installed
        if attn_implementation == "flash_attention_2":
            if not has_flash_attn():
                print("[WARNING] flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"

        load_pretrained_weights = qwenvl_config.get("load_pretrained_weights", True)
        if load_pretrained_weights:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                attn_implementation=attn_implementation,
                dtype=torch.bfloat16,
                ignore_mismatched_sizes=True, # resize image no longer needed? @TODO check bug
            )
        else:
            # Eval / from_pretrained(ckpt): build the architecture only. The
            # caller applies the training .pt with assign=True. Avoids an extra
            # 8 GiB HF shard load that was swapping the eval pod.
            hf_cfg = AutoConfig.from_pretrained(model_id)
            try:
                from accelerate import init_empty_weights
            except ImportError:
                init_empty_weights = None
            if init_empty_weights is not None:
                with init_empty_weights():
                    model = Qwen3VLForConditionalGeneration._from_config(
                        hf_cfg,
                        attn_implementation=attn_implementation,
                        dtype=torch.bfloat16,
                    )
            else:
                model = Qwen3VLForConditionalGeneration._from_config(
                    hf_cfg,
                    attn_implementation=attn_implementation,
                    dtype=torch.bfloat16,
                )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        # alin qwen3 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        # only for fast base model
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen2.5-VL backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def _maybe_apply_history_mrope(self, batch_inputs: dict, n_images_per_sample: int) -> dict:
        """WM-v1: rewrite the M-RoPE T channel so the leading history frames sit at
        T = 0, s, ..., (F-1)*s and the current views share T = F*s.

        Sets batch_inputs["position_ids"], which Qwen3VLForConditionalGeneration
        respects as-is (it only computes its own when position_ids is None).
        No-op when framework.working_memory is absent -> legacy behaviour.
        """
        wm = (self.config.framework.get("working_memory", None) or {})
        n_hist = int(wm.get("history_frames", 0) or 0)
        if n_hist == 0:
            return batch_inputs
        n_grid = int(batch_inputs["image_grid_thw"].shape[0])
        n_samples = int(batch_inputs["input_ids"].shape[0])
        assert n_grid == n_images_per_sample * n_samples, (
            f"working_memory layout broken: {n_grid} image grids for "
            f"{n_samples} samples x {n_images_per_sample} views"
        )
        # mm_token_type_ids exists only in newer transformers (5.x). Older versions
        # (e.g. the training image) neither return it from the processor nor accept
        # it in get_rope_index -- there, image spans are recovered from input_ids,
        # which is exactly what the old get_rope_index scans internally.
        mm_types = batch_inputs.get("mm_token_type_ids")
        if mm_types is None:
            input_ids = batch_inputs["input_ids"]
            mm_types = torch.zeros_like(input_ids)
            mm_types[(input_ids == IMAGE_TOKEN_INDEX) | (input_ids == VIDEO_TOKEN_INDEX)] = 1
        rope_fn = self.model.model.get_rope_index
        rope_kwargs = {
            "image_grid_thw": batch_inputs.get("image_grid_thw"),
            "video_grid_thw": batch_inputs.get("video_grid_thw"),
            "attention_mask": batch_inputs["attention_mask"],
        }
        if "mm_token_type_ids" in inspect.signature(rope_fn).parameters:
            rope_kwargs["mm_token_type_ids"] = mm_types
        position_ids, _ = rope_fn(batch_inputs["input_ids"], **rope_kwargs)
        batch_inputs["position_ids"] = apply_history_mrope(
            position_ids,
            mm_types,
            batch_inputs["attention_mask"],
            history_frames=n_hist,
            t_stride=int(wm.get("mrope_t_stride", 2)),
        )
        return batch_inputs

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
        """

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference

        batch_inputs = self.processor.apply_chat_template(
            messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )

        # if solutions, mask out the solution tokens in labels
        if solutions is not None:  #  here only for fast_tokenizer now.
            action_token_min = _ACTION_TOKEN_MIN  # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
            labels = batch_inputs["input_ids"].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    RuntimeWarning(
                        "action token are on in yout tokenizer, plz see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md."
                    )

            labels[labels == self.processor.tokenizer.pad_token_id] = -100  ## mask out pad tokens as well
            batch_inputs["labels"] = labels

        batch_inputs = self._maybe_apply_history_mrope(batch_inputs, n_images_per_sample=len(images[0]))
        return batch_inputs.to(self.model.device)


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    qwen_vl = _QWen3_VL_Interface(cfg)
    pass
