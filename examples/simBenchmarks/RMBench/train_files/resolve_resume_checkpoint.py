"""Resolve a local/EFS or S3 checkpoint to a weight file with a saved step.

Directories may be a run root or its checkpoints/ directory. S3 sources are
downloaded to an isolated pod-local directory; EFS files are read in place.
"""

import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile


PATTERN = re.compile(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)")


def select_checkpoint(source):
    path = Path(source).resolve()
    if path.is_dir():
        directory = path / "checkpoints" if (path / "checkpoints").is_dir() else path
        candidates = [p for p in directory.iterdir() if p.is_file() and PATTERN.fullmatch(p.name)]
    else:
        candidates = [path] if path.is_file() and PATTERN.fullmatch(path.name) else []
    if not candidates:
        raise ValueError(f"No steps_N checkpoint found at {source}; final_model alone has no resume step")
    return max(candidates, key=lambda p: (int(PATTERN.fullmatch(p.name)[1]), p.suffix == ".safetensors"))


def resolve_checkpoint(source, cache_root):
    if source.startswith("s3://"):
        Path(cache_root).mkdir(parents=True, exist_ok=True)
        cache = Path(tempfile.mkdtemp(prefix="resume-", dir=cache_root))
        if PATTERN.fullmatch(source.rsplit("/", 1)[-1]):
            target = cache / source.rsplit("/", 1)[-1]
            subprocess.run(["aws", "s3", "cp", source, str(target)], check=True, stdout=sys.stderr)
        else:
            target = cache
            subprocess.run([
                "aws", "s3", "sync", source, str(target), "--exclude", "*",
                "--include", "steps_*_model.safetensors", "--include", "steps_*_pytorch_model.pt",
                "--include", "checkpoints/steps_*_model.safetensors",
                "--include", "checkpoints/steps_*_pytorch_model.pt",
            ], check=True, stdout=sys.stderr)
        source = target
    return select_checkpoint(source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--cache-root", default="/local-ssd/rmbench_resume")
    args = parser.parse_args()
    print(resolve_checkpoint(args.source, args.cache_root))
