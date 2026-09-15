"""自动计算训练步数的统一入口（Epoch-based training-step budgeting）

Derives ``trainer.max_train_steps`` from ``trainer.num_train_epochs`` so the
training budget is invariant to per-device batch size, GPU count and
``data_mix`` -- important when comparing runs (e.g. working-memory vs
baseline), where an absolute step count would silently change how much data
each run actually sees.

Kept dependency-free (math + logging only) so it stays unit-testable without
the training image's heavy stack (deepspeed etc.).
"""

import logging
import math

logger = logging.getLogger(__name__)


def _dataset_num_samples(dataloader) -> int:
    """Number of unique samples backing *dataloader*.

    NOTE: ``LeRobotMixtureDataset.__len__`` is ``max(len_i / weight_i)`` -- the number
    of draws needed to cover the *largest* sub-dataset once -- which overstates the
    data actually available (405,876 vs 273,465 for ``libero_all``).  Sum the
    per-dataset lengths instead, so that "1 epoch" means "every frame seen once on
    average".  Sampling is with replacement, so this is an expectation, not an
    exact traversal.
    """
    ds = getattr(dataloader, "dataset", None)
    if ds is None:
        return len(dataloader)
    lengths = getattr(ds, "dataset_lengths", None)  # LeRobotMixtureDataset only
    if lengths is not None:
        return int(sum(lengths))
    return len(ds)


def resolve_max_train_steps(cfg, dataloader, accelerator) -> None:
    """Derive ``trainer.max_train_steps`` from ``trainer.num_train_epochs`` when set.

    Must be called BEFORE ``setup_optimizer_and_scheduler``, which reads
    ``cfg.trainer.max_train_steps`` to size the LR schedule.

    No-op when ``num_train_epochs`` is absent, so configs that specify an explicit
    ``max_train_steps`` keep working unchanged.
    """
    epochs = getattr(cfg.trainer, "num_train_epochs", None)
    if not epochs:
        return

    num_samples = _dataset_num_samples(dataloader)
    # Use the accelerator's *effective* accumulation (from the DeepSpeed/accelerate
    # config), never cfg.trainer.gradient_accumulation_steps -- nothing reads that key.
    global_batch_size = (
        cfg.datasets.vla_data.per_device_batch_size
        * accelerator.num_processes
        * accelerator.gradient_accumulation_steps
    )
    max_train_steps = math.ceil(float(epochs) * num_samples / global_batch_size)
    cfg.trainer.max_train_steps = max_train_steps

    logger.info(
        f"Derived max_train_steps={max_train_steps} from num_train_epochs={epochs} "
        f"(dataset_samples={num_samples}, global_batch_size={global_batch_size})"
    )
