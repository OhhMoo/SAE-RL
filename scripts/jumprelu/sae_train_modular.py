# %% [markdown]
# # Modular Qwen JumpReLU SAE training
#
# Run this file directly on a multi-GPU RunPod. Environment variables in main()
# select the layer, worker count, output path, and one-wave pilot mode.
# PPO rollouts are split by unique prompt so related completions stay together.

# %%
# Colab setup (safe to skip when the packages are already installed).
# %pip install -q transformers accelerate safetensors datasets tqdm

# %%
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import queue
import random
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class CacheConfig:
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    rollout_dataset_id: str = "tchalfpenny/qwen2.5-0.5b-gsm8k-rollouts"
    max_length: int = 768
    forward_batch_size: int = 16
    chunk_size_tokens: int = 100_000
    max_train_tokens: int = 1_000_000
    max_validation_tokens: int = 100_000
    max_test_tokens: int = 100_000
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 0
    truncate_after_hooked_layer: bool = True


@dataclass(frozen=True)
class TrainConfig:
    expansion_factor: int = 16
    l0_coefficient: float = 1e-4
    learning_rate: float = 3e-4
    batch_size: int = 4096
    steps: int = 5_000
    mixed_precision: str = "auto"
    fused_optimizer: bool = True
    compile_model: bool = False
    preload_activations: bool = True
    eval_every: int = 250
    max_validation_batches: int | None = None
    init_threshold: float = 0.03
    bandwidth: float = 0.1
    seed: int = 0


def load_qwen(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else (
        torch.float16 if use_cuda else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def get_decoder_layers(model) -> nn.ModuleList:
    """Return Qwen decoder blocks and fail clearly for an incompatible model."""
    try:
        return model.model.layers
    except AttributeError as exc:
        raise TypeError("Expected a Qwen-style model with model.model.layers") from exc


def validate_layer(model, layer: int) -> None:
    n_layers = len(get_decoder_layers(model))
    if not 0 <= layer < n_layers:
        raise ValueError(f"layer must be in [0, {n_layers - 1}], got {layer}")


def _prompt_key(prompt) -> str:
    """Create a stable key for either string or chat-message prompts."""
    return prompt if isinstance(prompt, str) else json.dumps(prompt, sort_keys=True)


def prepare_rollout_splits(config: CacheConfig) -> dict[str, Dataset]:
    """Load PPO rollouts and keep all completions for a prompt in one split."""
    rollouts = load_dataset(
        config.rollout_dataset_id,
        data_files={"train": "train.jsonl"},
        split="train",
    )
    required = {"prompt", "completion", "full_text"}
    missing = required.difference(rollouts.column_names)
    if missing:
        raise ValueError(
            f"Rollout dataset is missing columns {sorted(missing)}; "
            f"found {rollouts.column_names}"
        )

    row_keys = [_prompt_key(prompt) for prompt in rollouts["prompt"]]
    unique_keys = list(dict.fromkeys(row_keys))
    if len(unique_keys) < 3:
        raise ValueError("Need at least three unique prompts for train/validation/test")
    random.Random(config.seed).shuffle(unique_keys)

    n_test = max(1, round(len(unique_keys) * config.test_fraction))
    n_validation = max(1, round(len(unique_keys) * config.validation_fraction))
    if n_test + n_validation >= len(unique_keys):
        raise ValueError("validation_fraction + test_fraction leaves no training prompts")
    test_keys = set(unique_keys[:n_test])
    validation_keys = set(unique_keys[n_test : n_test + n_validation])

    indices = {"train": [], "validation": [], "test": []}
    for index, key in enumerate(row_keys):
        split = "test" if key in test_keys else (
            "validation" if key in validation_keys else "train"
        )
        indices[split].append(index)
    return {split: rollouts.select(rows) for split, rows in indices.items()}


def format_rollout_example(example: dict) -> str:
    """Use the exact prompt+policy-completion text stored during PPO rollout."""
    full_text = example["full_text"]
    if not isinstance(full_text, str) or not full_text:
        raise ValueError("Each rollout must contain a non-empty string full_text")
    return full_text


def _load_tensor_dict(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Older Colab PyTorch.
        return torch.load(path, map_location="cpu")


@torch.inference_mode()
def cache_activations(
    model,
    tokenizer,
    dataset: Dataset,
    layer: int,
    split: str,
    cache_root: str | Path,
    config: CacheConfig,
    max_tokens: int,
    overwrite: bool = False,
) -> Path:
    """Cache unpadded post-block residual activations in bounded CPU chunks."""
    validate_layer(model, layer)
    split_dir = Path(cache_root) / f"layer_{layer:02d}" / split
    manifest_path = split_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        manifest = json.loads(manifest_path.read_text())
        cached_dataset = manifest.get("cache_config", {}).get("rollout_dataset_id")
        if (
            manifest["layer"] != layer
            or manifest["model_id"] != config.model_id
            or cached_dataset != config.rollout_dataset_id
        ):
            raise ValueError(f"Incompatible cache already exists at {split_dir}")
        return split_dir

    split_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for old_chunk in split_dir.glob("chunk_*.pt"):
            old_chunk.unlink()

    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden.detach())

    decoder_layers = get_decoder_layers(model)
    handle = decoder_layers[layer].register_forward_hook(hook)
    original_layers = None
    if config.truncate_after_hooked_layer:
        # Later decoder blocks cannot affect a post-block activation.
        original_layers = model.model.layers
        model.model.layers = nn.ModuleList(list(original_layers[: layer + 1]))
    input_device = model.get_input_embeddings().weight.device
    buffers: dict[str, list[torch.Tensor]] = {
        "acts": [], "tokens": [], "text_ids": [], "positions": []
    }
    chunks: list[dict] = []
    buffered_tokens = 0
    total_tokens = 0

    def flush() -> None:
        nonlocal buffered_tokens
        if not buffers["acts"]:
            return
        payload = {key: torch.cat(value, dim=0) for key, value in buffers.items()}
        payload.update({"layer": layer, "hook": "resid_post", "split": split})
        chunk_name = f"chunk_{len(chunks):04d}.pt"
        torch.save(payload, split_dir / chunk_name)
        chunks.append({"file": chunk_name, "tokens": int(payload["acts"].shape[0])})
        for value in buffers.values():
            value.clear()
        buffered_tokens = 0

    try:
        starts = range(0, len(dataset), config.forward_batch_size)
        for start in tqdm(starts, desc=f"cache layer {layer} {split}"):
            if total_tokens >= max_tokens:
                break
            stop = min(start + config.forward_batch_size, len(dataset))
            rows = [dataset[i] for i in range(start, stop)]
            texts = [format_rollout_example(row) for row in rows]
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_length,
            )
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            captured.clear()
            # Call the base transformer to avoid allocating full-vocabulary logits.
            model.model(**encoded, use_cache=False, return_dict=True)
            if len(captured) != 1:
                raise RuntimeError(f"Expected one hook call, observed {len(captured)}")

            mask = encoded["attention_mask"].bool()
            hidden = captured[0]
            hidden_mask = mask.to(hidden.device)
            batch_positions = torch.arange(hidden.shape[1], device=hidden.device)
            batch_positions = batch_positions.expand(hidden.shape[0], -1)
            text_ids = torch.arange(start, stop, device=hidden.device)[:, None]
            text_ids = text_ids.expand(hidden.shape[0], hidden.shape[1])

            acts = hidden[hidden_mask]
            token_ids = encoded["input_ids"][mask]
            positions = batch_positions[hidden_mask]
            example_ids = text_ids[hidden_mask]
            take = min(acts.shape[0], max_tokens - total_tokens)
            if take <= 0:
                break

            buffers["acts"].append(acts[:take].to(torch.bfloat16).cpu())
            buffers["tokens"].append(token_ids[:take].cpu())
            buffers["positions"].append(positions[:take].cpu())
            buffers["text_ids"].append(example_ids[:take].cpu())
            buffered_tokens += take
            total_tokens += take
            if buffered_tokens >= config.chunk_size_tokens:
                flush()
        flush()
    finally:
        handle.remove()
        if original_layers is not None:
            model.model.layers = original_layers

    if not chunks:
        raise RuntimeError(f"No activations were cached for {split}")
    first = _load_tensor_dict(split_dir / chunks[0]["file"])
    manifest = {
        "model_id": config.model_id,
        "layer": layer,
        "hook": "resid_post",
        "split": split,
        "d_in": int(first["acts"].shape[1]),
        "total_tokens": total_tokens,
        "chunks": chunks,
        "cache_config": asdict(config),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return split_dir


def read_manifest(cache_dir: str | Path) -> dict:
    return json.loads((Path(cache_dir) / "manifest.json").read_text())


def compute_train_normalization(cache_dir: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a per-coordinate mean and one global RMS scale, streaming from disk."""
    cache_dir = Path(cache_dir)
    manifest = read_manifest(cache_dir)
    total = 0
    sum_x = torch.zeros(manifest["d_in"], dtype=torch.float64)
    sum_x2 = torch.tensor(0.0, dtype=torch.float64)
    for chunk_info in tqdm(manifest["chunks"], desc="train normalization"):
        x = _load_tensor_dict(cache_dir / chunk_info["file"])["acts"].double()
        total += x.shape[0]
        sum_x += x.sum(dim=0)
        sum_x2 += x.square().sum()
    mean = sum_x / total
    variance = (sum_x2 - total * mean.square().sum()) / (total * mean.numel())
    scale = variance.clamp_min(1e-12).sqrt()
    return mean.float(), scale.float()


def load_activation_tensor(
    cache_dir: str | Path,
    device: torch.device,
) -> torch.Tensor:
    """Load one split once onto a GPU to remove repeated cache-disk reads."""
    cache_dir = Path(cache_dir)
    manifest = read_manifest(cache_dir)
    activations = torch.empty(
        (manifest["total_tokens"], manifest["d_in"]),
        dtype=torch.bfloat16,
        device=device,
    )
    offset = 0
    for chunk_info in tqdm(
        manifest["chunks"], desc=f"preload {cache_dir.name} on {device}"
    ):
        chunk = _load_tensor_dict(cache_dir / chunk_info["file"])["acts"]
        stop = offset + chunk.shape[0]
        activations[offset:stop].copy_(chunk, non_blocking=False)
        offset = stop
    if offset != activations.shape[0]:
        raise RuntimeError(
            f"Manifest expected {activations.shape[0]} tokens, loaded {offset}"
        )
    return activations


def activation_batches(
    cache_dir: str | Path,
    batch_size: int,
    mean: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
    *,
    shuffle: bool,
    repeat: bool,
    seed: int,
    activations: torch.Tensor | None = None,
) -> Iterator[torch.Tensor]:
    """Yield normalized batches from a preloaded tensor or streamed chunks."""
    mean = mean.to(device)
    scale = scale.to(device)
    if activations is not None:
        if activations.device != device:
            raise ValueError("Preloaded activations must be on the training device")
        generator = torch.Generator(device=device).manual_seed(seed)
        while True:
            indices = (
                torch.randperm(activations.shape[0], generator=generator, device=device)
                if shuffle
                else torch.arange(activations.shape[0], device=device)
            )
            for start in range(0, activations.shape[0], batch_size):
                batch_indices = indices[start : start + batch_size]
                if repeat and batch_indices.numel() < batch_size:
                    continue
                batch = activations[batch_indices].float()
                yield (batch - mean) / scale
            if not repeat:
                return

    cache_dir = Path(cache_dir)
    manifest = read_manifest(cache_dir)
    generator = torch.Generator().manual_seed(seed)
    paths = [cache_dir / info["file"] for info in manifest["chunks"]]
    while True:
        order = (
            torch.randperm(len(paths), generator=generator).tolist()
            if shuffle
            else range(len(paths))
        )
        for path_index in order:
            x = _load_tensor_dict(paths[path_index])["acts"]
            indices = (
                torch.randperm(x.shape[0], generator=generator)
                if shuffle
                else torch.arange(x.shape[0])
            )
            for start in range(0, x.shape[0], batch_size):
                batch_indices = indices[start : start + batch_size]
                if repeat and batch_indices.numel() < batch_size:
                    continue
                batch = x[batch_indices].float().to(device, non_blocking=True)
                yield (batch - mean) / scale
        if not repeat:
            return


def resolve_mixed_precision(
    setting: str,
    device: torch.device,
) -> torch.dtype | None:
    setting = setting.lower()
    if setting not in {"auto", "bf16", "fp16", "fp32"}:
        raise ValueError("mixed_precision must be auto, bf16, fp16, or fp32")
    if device.type != "cuda" or setting == "fp32":
        return None
    if setting == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if setting == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support BF16; use fp16 or auto")
        return torch.bfloat16
    return torch.float16


# %%
class JumpReLUSAE(nn.Module):
    """Hard JumpReLU forward pass with a sigmoid straight-through surrogate."""

    def __init__(self, d_in: int, d_sae: int, init_threshold: float, bandwidth: float):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.bandwidth = bandwidth
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in) / math.sqrt(d_in))
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.log_threshold = nn.Parameter(torch.full((d_sae,), math.log(init_threshold)))
        self.normalize_decoder()
        with torch.no_grad():
            self.W_enc.copy_(self.W_dec.T)

    @property
    def threshold(self) -> torch.Tensor:
        return self.log_threshold.exp()

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def remove_decoder_parallel_gradient(self) -> None:
        if self.W_dec.grad is None:
            return
        parallel = (self.W_dec.grad * self.W_dec).sum(dim=1, keepdim=True)
        self.W_dec.grad.sub_(parallel * self.W_dec)

    def forward(self, x: torch.Tensor):
        pre = x @ self.W_enc + self.b_enc
        hard_mask = (pre > self.threshold).to(pre.dtype)
        soft_mask = torch.sigmoid((pre - self.threshold) / self.bandwidth)
        straight_through_mask = hard_mask.detach() - soft_mask.detach() + soft_mask
        features = pre * straight_through_mask
        reconstruction = features @ self.W_dec + self.b_dec
        return reconstruction, features, hard_mask, soft_mask


@torch.inference_mode()
def evaluate_sae(
    sae: JumpReLUSAE,
    cache_dir: str | Path,
    mean: torch.Tensor,
    scale: torch.Tensor,
    batch_size: int,
    l0_coefficient: float,
    device: torch.device,
    max_batches: int | None = None,
    mixed_precision: str = "auto",
    activations: torch.Tensor | None = None,
) -> dict[str, float]:
    """Evaluate the deploy-time hard SAE; this is the model-selection objective."""
    sae.eval()
    amp_dtype = resolve_mixed_precision(mixed_precision, device)
    n_tokens = 0
    squared_error = 0.0
    x_sum = torch.zeros(sae.d_in, dtype=torch.float64)
    x_squared_sum = 0.0
    cosine_sum = 0.0
    l0_sum = 0.0
    ever_active = torch.zeros(sae.d_sae, dtype=torch.bool, device=device)
    batches = activation_batches(
        cache_dir,
        batch_size,
        mean,
        scale,
        device,
        shuffle=False,
        repeat=False,
        seed=0,
        activations=activations,
    )
    for batch_number, x in enumerate(batches):
        if max_batches is not None and batch_number >= max_batches:
            break
        amp_context = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_dtype is not None
            else nullcontext()
        )
        with amp_context:
            x_hat, _features, hard_mask, _soft_mask = sae(x)
        n_tokens += x.shape[0]
        x_float = x.float()
        squared_error += (x_hat.float() - x_float).square().sum().item()
        x_sum += x_float.sum(dim=0).cpu().double()
        x_squared_sum += x_float.square().sum().item()
        cosine_sum += F.cosine_similarity(x_hat.float(), x_float, dim=-1).sum().item()
        l0_sum += hard_mask.sum().item()
        ever_active |= hard_mask.bool().any(dim=0)
    if n_tokens == 0:
        raise RuntimeError("Evaluation cache produced no batches")
    mse = squared_error / (n_tokens * sae.d_in)
    hard_l0 = l0_sum / n_tokens
    total_variance = max(x_squared_sum - x_sum.square().sum().item() / n_tokens, 1e-12)
    metrics = {
        "hard_objective": mse + l0_coefficient * hard_l0,
        "mse": mse,
        "explained_variance": 1.0 - squared_error / total_variance,
        "cosine_similarity": cosine_sum / n_tokens,
        "hard_l0": hard_l0,
        "dead_feature_fraction": (~ever_active).float().mean().item(),
        "tokens": n_tokens,
    }
    sae.train()
    return metrics


def train_sae(
    train_cache: str | Path,
    validation_cache: str | Path,
    output_path: str | Path,
    layer: int,
    config: TrainConfig,
    normalization: tuple[torch.Tensor, torch.Tensor] | None = None,
    device: torch.device | str | None = None,
    train_activations: torch.Tensor | None = None,
    validation_activations: torch.Tensor | None = None,
) -> tuple[JumpReLUSAE, list[dict]]:
    # Seed CPU initialization without touching every visible CUDA device.
    torch.random.default_generator.manual_seed(config.seed)
    random.seed(config.seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(config.seed)

    manifest = read_manifest(train_cache)
    if manifest["layer"] != layer or read_manifest(validation_cache)["layer"] != layer:
        raise ValueError("Train/validation cache layer does not match requested layer")

    mean, scale = normalization or compute_train_normalization(train_cache)
    if config.preload_activations and device.type == "cuda":
        if train_activations is None:
            train_activations = load_activation_tensor(train_cache, device)
        if validation_activations is None:
            validation_activations = load_activation_tensor(validation_cache, device)

    sae = JumpReLUSAE(
        d_in=manifest["d_in"],
        d_sae=manifest["d_in"] * config.expansion_factor,
        init_threshold=config.init_threshold,
        bandwidth=config.bandwidth,
    ).to(device)
    if config.compile_model:
        if not hasattr(sae, "compile"):
            raise RuntimeError("compile_model requires PyTorch 2.1 or newer")
        sae.compile(mode="reduce-overhead")

    optimizer_kwargs = {"lr": config.learning_rate}
    if config.fused_optimizer and device.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.Adam(sae.parameters(), **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.Adam(sae.parameters(), **optimizer_kwargs)

    amp_dtype = resolve_mixed_precision(config.mixed_precision, device)
    use_scaler = amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    train_batches = activation_batches(
        train_cache,
        config.batch_size,
        mean,
        scale,
        device,
        shuffle=True,
        repeat=True,
        seed=config.seed,
        activations=train_activations,
    )
    history: list[dict] = []
    best_validation = math.inf
    best_state = None
    gpu_label = f" gpu {device.index}" if device.type == "cuda" else ""

    for step in tqdm(
        range(1, config.steps + 1),
        desc=f"train layer {layer}{gpu_label}",
        position=(device.index or 0) if device.type == "cuda" else 0,
    ):
        x = next(train_batches)
        optimizer.zero_grad(set_to_none=True)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_dtype is not None
            else nullcontext()
        )
        with amp_context:
            x_hat, _features, hard_mask, soft_mask = sae(x)
            reconstruction_mse = F.mse_loss(x_hat.float(), x.float())
            soft_l0 = soft_mask.float().sum(dim=-1).mean()
            loss = reconstruction_mse + config.l0_coefficient * soft_l0

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        sae.remove_decoder_parallel_gradient()
        scaler.step(optimizer)
        scaler.update()
        sae.normalize_decoder()

        if step == 1 or step % config.eval_every == 0 or step == config.steps:
            validation = evaluate_sae(
                sae,
                validation_cache,
                mean,
                scale,
                config.batch_size,
                config.l0_coefficient,
                device,
                config.max_validation_batches,
                mixed_precision=config.mixed_precision,
                activations=validation_activations,
            )
            record = {
                "step": step,
                "train_surrogate_objective": loss.item(),
                "train_mse": reconstruction_mse.item(),
                "train_soft_l0": soft_l0.item(),
                "train_hard_l0": hard_mask.sum(dim=-1).float().mean().item(),
                **{f"validation_{key}": value for key, value in validation.items()},
            }
            history.append(record)
            print(json.dumps(record, indent=2))
            if validation["hard_objective"] < best_validation:
                best_validation = validation["hard_objective"]
                # Keep the best snapshot on-device during training. Copying a
                # full state to CPU at every improvement stalls all CUDA work.
                best_state = {
                    key: value.detach().clone()
                    for key, value in sae.state_dict().items()
                }

    if best_state is None:
        raise RuntimeError("Training ended without a validation checkpoint")
    sae.load_state_dict(best_state)
    checkpoint_state = {
        key: value.detach().cpu() for key, value in best_state.items()
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sae_state_dict": checkpoint_state,
            "layer": layer,
            "hook": "resid_post",
            "d_in": sae.d_in,
            "d_sae": sae.d_sae,
            "normalization_mean": mean,
            "normalization_scale": scale,
            "train_config": asdict(config),
            "cache_manifest": manifest,
            "history": history,
            "best_validation_hard_objective": best_validation,
        },
        output_path.with_suffix(output_path.suffix + ".tmp"),
    )
    os.replace(output_path.with_suffix(output_path.suffix + ".tmp"), output_path)
    return sae, history


# %%
def train_layer_pipeline(
    layer: int,
    work_dir: str | Path = "/content/drive/MyDrive/qwen_sae",
    cache_config: CacheConfig | None = None,
    train_config: TrainConfig | None = None,
    run_test: bool = True,
) -> dict:
    """Cache, train, select on validation, and optionally evaluate test once."""
    cache_config = cache_config or CacheConfig()
    train_config = train_config or TrainConfig()
    work_dir = Path(work_dir)
    model, tokenizer = load_qwen(cache_config.model_id)
    validate_layer(model, layer)
    corpora = prepare_rollout_splits(cache_config)
    limits = {
        "train": cache_config.max_train_tokens,
        "validation": cache_config.max_validation_tokens,
        "test": cache_config.max_test_tokens,
    }
    cache_dirs = {}
    dataset_slug = cache_config.rollout_dataset_id.replace("/", "--")
    cache_root = work_dir / "activations" / dataset_slug
    requested_splits = ["train", "validation"] + (["test"] if run_test else [])
    for split in requested_splits:
        cache_dirs[split] = cache_activations(
            model, tokenizer, corpora[split], layer, split,
            cache_root, cache_config, limits[split],
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    checkpoint_path = work_dir / "saes" / dataset_slug / f"jumprelu_layer_{layer:02d}.pt"
    sae, history = train_sae(
        cache_dirs["train"], cache_dirs["validation"], checkpoint_path,
        layer, train_config,
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "layer": layer,
        "best_validation_hard_objective": min(
            row["validation_hard_objective"] for row in history
        ),
    }
    if run_test:
        checkpoint = _load_tensor_dict(checkpoint_path)
        device = next(sae.parameters()).device
        test_metrics = evaluate_sae(
            sae,
            cache_dirs["test"],
            checkpoint["normalization_mean"],
            checkpoint["normalization_scale"],
            train_config.batch_size,
            train_config.l0_coefficient,
            device,
        )
        result["test"] = test_metrics
        checkpoint["test_metrics"] = test_metrics
        torch.save(checkpoint, checkpoint_path)
    print(json.dumps(result, indent=2))
    return result


def _float_tag(value: float) -> str:
    return f"{value:.8g}".replace(".", "p").replace("-", "m").replace("+", "")


def _sweep_result_from_history(
    job: dict,
    checkpoint_path: str | Path,
    history: list[dict],
    selection_l0_coefficient: float,
) -> dict:
    best_record = min(history, key=lambda row: row["validation_hard_objective"])
    result = {
        "run_number": job["run_number"],
        "layer": job["layer"],
        "l0_coefficient": job["l0_coefficient"],
        "learning_rate": job["learning_rate"],
        "checkpoint": str(checkpoint_path),
        "best_step": best_record["step"],
        **{
            key: value
            for key, value in best_record.items()
            if key.startswith("validation_")
        },
    }
    # Every run must use the same selection penalty when comparing different
    # training L0 coefficients.
    result["validation_selection_objective"] = (
        result["validation_mse"]
        + selection_l0_coefficient * result["validation_hard_l0"]
    )
    return result


def _run_sweep_job(
    job: dict,
    train_cache: str,
    validation_cache: str,
    normalization: tuple[torch.Tensor, torch.Tensor],
    base_train_config: TrainConfig,
    device: torch.device,
    selection_l0_coefficient: float,
    train_activations: torch.Tensor | None,
    validation_activations: torch.Tensor | None,
) -> dict:
    run_config = replace(
        base_train_config,
        l0_coefficient=float(job["l0_coefficient"]),
        learning_rate=float(job["learning_rate"]),
    )
    checkpoint_path = Path(job["checkpoint"])
    if checkpoint_path.exists():
        checkpoint = _load_tensor_dict(checkpoint_path)
        if checkpoint.get("train_config") != asdict(run_config):
            raise RuntimeError(
                f"Existing checkpoint has a different config: {checkpoint_path}"
            )
        history = checkpoint["history"]
        print(f"GPU {device.index}: reusing completed {checkpoint_path.name}")
    else:
        print(
            f"GPU {device.index}: run {job['run_number']} "
            f"L0={job['l0_coefficient']:g} LR={job['learning_rate']:g}"
        )
        sae, history = train_sae(
            train_cache,
            validation_cache,
            checkpoint_path,
            job["layer"],
            run_config,
            normalization=normalization,
            device=device,
            train_activations=train_activations,
            validation_activations=validation_activations,
        )
        del sae
        torch.cuda.empty_cache()

    result = _sweep_result_from_history(
        job, checkpoint_path, history, selection_l0_coefficient
    )
    result_path = checkpoint_path.with_suffix(".result.json")
    result_path.write_text(json.dumps(result, indent=2))
    return result


def _gpu_sweep_worker(
    gpu_index: int,
    job_queue,
    result_queue,
    train_cache: str,
    validation_cache: str,
    normalization: tuple[torch.Tensor, torch.Tensor],
    base_train_config_dict: dict,
    selection_l0_coefficient: float,
) -> None:
    try:
        device = torch.device(f"cuda:{gpu_index}")
        torch.cuda.set_device(device)
        base_train_config = TrainConfig(**base_train_config_dict)
        train_activations = None
        validation_activations = None
        if base_train_config.preload_activations:
            train_activations = load_activation_tensor(train_cache, device)
            validation_activations = load_activation_tensor(validation_cache, device)

        while True:
            job = job_queue.get()
            if job is None:
                break
            try:
                result = _run_sweep_job(
                    job,
                    train_cache,
                    validation_cache,
                    normalization,
                    base_train_config,
                    device,
                    selection_l0_coefficient,
                    train_activations,
                    validation_activations,
                )
                result_queue.put({"ok": True, "result": result})
            except Exception:
                result_queue.put(
                    {
                        "ok": False,
                        "job": job,
                        "traceback": traceback.format_exc(),
                    }
                )
    except Exception:
        result_queue.put(
            {
                "ok": False,
                "job": None,
                "traceback": traceback.format_exc(),
            }
        )


def sweep_layer_pipeline(
    layer: int,
    l0_coefficients: list[float],
    learning_rates: list[float],
    work_dir: str | Path = "/workspace/qwen_sae",
    cache_config: CacheConfig | None = None,
    base_train_config: TrainConfig | None = None,
    test_best_only: bool = True,
    parallel_workers: int | None = None,
    selection_l0_coefficient: float | None = None,
) -> list[dict]:
    """Cache one layer, then distribute independent sweep trials over all GPUs."""
    if not l0_coefficients or not learning_rates:
        raise ValueError("l0_coefficients and learning_rates must both be non-empty")
    if any(value < 0 for value in l0_coefficients):
        raise ValueError("L0 coefficients must be non-negative")
    if any(value <= 0 for value in learning_rates):
        raise ValueError("Learning rates must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("The parallel sweep requires at least one CUDA GPU")

    cache_config = cache_config or CacheConfig()
    base_train_config = base_train_config or TrainConfig()
    selection_l0_coefficient = (
        base_train_config.l0_coefficient
        if selection_l0_coefficient is None
        else float(selection_l0_coefficient)
    )
    work_dir = Path(work_dir)
    model, tokenizer = load_qwen(cache_config.model_id)
    validate_layer(model, layer)
    corpora = prepare_rollout_splits(cache_config)
    limits = {
        "train": cache_config.max_train_tokens,
        "validation": cache_config.max_validation_tokens,
        "test": cache_config.max_test_tokens,
    }
    dataset_slug = cache_config.rollout_dataset_id.replace("/", "--")
    cache_root = work_dir / "activations" / dataset_slug
    requested_splits = ["train", "validation"] + (["test"] if test_best_only else [])
    cache_dirs = {
        split: cache_activations(
            model,
            tokenizer,
            corpora[split],
            layer,
            split,
            cache_root,
            cache_config,
            limits[split],
        )
        for split in requested_splits
    }
    del model
    torch.cuda.empty_cache()

    normalization = compute_train_normalization(cache_dirs["train"])
    sweep_dir = work_dir / "saes" / dataset_slug / f"layer_{layer:02d}_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    combinations = list(product(l0_coefficients, learning_rates))
    jobs = []
    for run_number, (l0_coefficient, learning_rate) in enumerate(combinations, start=1):
        checkpoint_path = sweep_dir / (
            f"sae_l0_{_float_tag(l0_coefficient)}"
            f"_lr_{_float_tag(learning_rate)}.pt"
        )
        jobs.append(
            {
                "run_number": run_number,
                "layer": layer,
                "l0_coefficient": float(l0_coefficient),
                "learning_rate": float(learning_rate),
                "checkpoint": str(checkpoint_path),
            }
        )

    available_gpus = torch.cuda.device_count()
    worker_count = min(parallel_workers or available_gpus, available_gpus, len(jobs))
    if worker_count < 1:
        raise RuntimeError("No CUDA workers are available")
    print(
        f"Launching {len(jobs)} trials on {worker_count} GPUs; "
        f"selection L0={selection_l0_coefficient:g}"
    )

    context = mp.get_context("spawn")
    job_queue = context.Queue()
    result_queue = context.Queue()
    for job in jobs:
        job_queue.put(job)
    for _ in range(worker_count):
        job_queue.put(None)

    workers = [
        context.Process(
            target=_gpu_sweep_worker,
            args=(
                gpu_index,
                job_queue,
                result_queue,
                str(cache_dirs["train"]),
                str(cache_dirs["validation"]),
                normalization,
                asdict(base_train_config),
                selection_l0_coefficient,
            ),
        )
        for gpu_index in range(worker_count)
    ]
    for worker in workers:
        worker.start()

    results: list[dict] = []
    errors: list[dict] = []
    messages_received = 0
    while messages_received < len(jobs):
        try:
            message = result_queue.get(timeout=5)
        except queue.Empty:
            if not any(worker.is_alive() for worker in workers):
                break
            continue
        messages_received += 1
        if message["ok"]:
            results.append(message["result"])
            partial = sorted(
                results, key=lambda row: row["validation_selection_objective"]
            )
            (sweep_dir / "sweep_results.json").write_text(
                json.dumps(partial, indent=2)
            )
            print(f"Completed {len(results)}/{len(jobs)} trials")
        else:
            errors.append(message)

    for worker in workers:
        worker.join()
    bad_exit_codes = [worker.exitcode for worker in workers if worker.exitcode != 0]
    if errors or bad_exit_codes or len(results) != len(jobs):
        details = "\n\n".join(error["traceback"] for error in errors)
        raise RuntimeError(
            f"Sweep failed: {len(results)}/{len(jobs)} trials completed; "
            f"worker exit codes={bad_exit_codes}\n{details}"
        )

    results.sort(key=lambda row: row["validation_selection_objective"])
    if test_best_only:
        winner = results[0]
        checkpoint_path = Path(winner["checkpoint"])
        checkpoint = _load_tensor_dict(checkpoint_path)
        winner_config = TrainConfig(**checkpoint["train_config"])
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        winner_sae = JumpReLUSAE(
            d_in=checkpoint["d_in"],
            d_sae=checkpoint["d_sae"],
            init_threshold=winner_config.init_threshold,
            bandwidth=winner_config.bandwidth,
        ).to(device)
        winner_sae.load_state_dict(checkpoint["sae_state_dict"])
        test_metrics = evaluate_sae(
            winner_sae,
            cache_dirs["test"],
            checkpoint["normalization_mean"],
            checkpoint["normalization_scale"],
            winner_config.batch_size,
            selection_l0_coefficient,
            device,
            mixed_precision=winner_config.mixed_precision,
        )
        winner["test_metrics"] = test_metrics
        checkpoint["test_metrics"] = test_metrics
        temporary_path = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, checkpoint_path)

    (sweep_dir / "sweep_results.json").write_text(json.dumps(results, indent=2))
    print(f"Completed {len(results)} runs. Best checkpoint: {results[0]['checkpoint']}")
    return results


# %% [markdown]
# ## Sweep one layer
#
# Every L0 coefficient is paired with every learning rate. Validation selects the
# winner; test is evaluated once for that winner and is not used for tuning.

# %%
# If this dataset is private, authenticate once before running the pipeline:
# from huggingface_hub import notebook_login
# notebook_login()

def _environment_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def main() -> None:
    layer = int(os.environ.get("QWEN_SAE_LAYER", "6"))
    base_work_dir = Path(os.environ.get("QWEN_SAE_WORK_DIR", "/workspace/qwen_sae"))
    parallel_workers = int(os.environ.get("QWEN_SAE_WORKERS", "8"))
    pilot = _environment_flag("QWEN_SAE_PILOT")

    if pilot:
        # Exactly eight representative jobs: one wave on an 8-GPU pod.
        l0_coefficients = [3e-5, 1e-4, 3e-4, 1e-3]
        learning_rates = [1e-4, 3e-4]
        default_steps = 500
        default_train_tokens = 250_000
        default_eval_tokens = 25_000
        work_dir = base_work_dir / "pilot"
    else:
        l0_coefficients = [
            1e-5,
            3e-5,
            1e-4,
            3e-4,
            1e-3,
            3e-3,
            1e-2,
            3e-2,
        ]
        learning_rates = [
            3e-5,
            5e-5,
            1e-4,
            2e-4,
            3e-4,
            5e-4,
            7e-4,
            1e-3,
        ]
        default_steps = 5_000
        default_train_tokens = 1_000_000
        default_eval_tokens = 100_000
        work_dir = base_work_dir

    steps = int(os.environ.get("QWEN_SAE_STEPS", str(default_steps)))
    train_tokens = int(
        os.environ.get("QWEN_SAE_TRAIN_TOKENS", str(default_train_tokens))
    )
    validation_tokens = int(
        os.environ.get("QWEN_SAE_VALIDATION_TOKENS", str(default_eval_tokens))
    )
    test_tokens = int(
        os.environ.get("QWEN_SAE_TEST_TOKENS", str(default_eval_tokens))
    )
    eval_every = int(os.environ.get("QWEN_SAE_EVAL_EVERY", "250"))

    sweep_results = sweep_layer_pipeline(
        layer=layer,
        l0_coefficients=l0_coefficients,
        learning_rates=learning_rates,
        work_dir=work_dir,
        cache_config=CacheConfig(
            max_train_tokens=train_tokens,
            max_validation_tokens=validation_tokens,
            max_test_tokens=test_tokens,
        ),
        base_train_config=TrainConfig(
            expansion_factor=16,
            batch_size=4096,
            steps=steps,
            eval_every=eval_every,
            mixed_precision="bf16",
            fused_optimizer=True,
            compile_model=False,
            preload_activations=True,
            seed=0,
        ),
        test_best_only=True,
        parallel_workers=parallel_workers,
        selection_l0_coefficient=1e-4,
    )
    assert len(sweep_results) == len(l0_coefficients) * len(learning_rates)


if __name__ == "__main__":
    main()
