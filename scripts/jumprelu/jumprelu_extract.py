"""GSM8K loading and residual-stream activation extraction via forward hooks.

Used to pull activations at layers not covered by the cached activation
dataset (which only has layers 6, 12, 18, 23).
Reproduces the train/val split ratio described in that dataset's README
(356,596 / 89,148, seed=0) as closely as possible -- see the NOTE below, this
is a best-effort match, not a confirmed-identical split.
"""
from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


DEFAULT_TOKEN_BUDGET = 445_744  # per team README: ~2M raw tokens / (stage, layer), real positions only


class GSM8KActivationDataset(Dataset):
    """Tokenizes GSM8K (question, answer) pairs using the model's chat template."""

    def __init__(self, hf_dataset, tokenizer, max_seq_len: int = 512):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        messages = [{"role": "user", "content": example["question"]}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt + example["answer"]
        enc = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return enc["input_ids"][0], enc["attention_mask"][0]


def load_gsm8k_split(seed: int = 0, train_frac_from: tuple = (356_596, 89_148)):
    """Reproduces the team's train/val split ratio on GSM8K train.

    NOTE: the team's README reports 356,596 / 89,148 rows for train/val, which
    is far larger than raw GSM8K train (7,473 rows) -- this almost certainly
    refers to token positions or an expanded sampling scheme, not raw dataset
    rows. This function reproduces the *ratio* (~80/20) via shuffle-and-slice
    on the raw dataset; it is not confirmed to reproduce the exact same rows
    as the original pipeline. Confirm with the team if row-level alignment
    matters for your use case.
    """
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    gsm8k_shuffled = gsm8k.shuffle(seed=seed)

    n_total = len(gsm8k_shuffled)
    n_train_rows, n_val_rows = train_frac_from
    n_train = int(n_total * n_train_rows / (n_train_rows + n_val_rows))

    train_ds = gsm8k_shuffled.select(range(n_train))
    val_ds = gsm8k_shuffled.select(range(n_train, n_total))
    return train_ds, val_ds


def build_val_loader(
    val_ds,
    model_name: str,
    max_seq_len: int = 512,
    batch_size: int = 16,
    tokenizer: Optional[AutoTokenizer] = None,
):
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = GSM8KActivationDataset(val_ds, tokenizer, max_seq_len=max_seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False), tokenizer


def extract_layer_activations(
    model,
    layer_idx: int,
    val_loader: DataLoader,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    device: str = "cuda",
) -> torch.Tensor:
    """Extracts real-token-only residual-stream activations at model.model.layers[layer_idx].

    Returns a (token_budget, d_model) tensor on CPU. Assumes an HF
    AutoModelForCausalLM-style model with a `.model.layers` list of decoder
    blocks (true for the Qwen2 architecture used in this project).
    """
    captured = []

    def hook_fn(module, module_input, module_output):
        hs = module_output[0] if isinstance(module_output, tuple) else module_output
        captured.append(hs.detach().cpu())

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    all_real_token_acts = []
    collected = 0

    with torch.no_grad():
        for input_ids, attention_mask in tqdm(val_loader, desc=f"Extracting layer {layer_idx}"):
            input_ids = input_ids.to(device)
            attention_mask_gpu = attention_mask.to(device)

            captured.clear()
            _ = model(input_ids, attention_mask=attention_mask_gpu)
            batch_acts = captured[0]

            mask_cpu = attention_mask.bool()
            real_acts = batch_acts[mask_cpu]

            all_real_token_acts.append(real_acts)
            collected += real_acts.shape[0]

            if collected >= token_budget:
                break

    handle.remove()
    return torch.cat(all_real_token_acts, dim=0)[:token_budget]


def normalize_activations(raw_acts: torch.Tensor, train_frac: float = 0.8, seed: int = 0):
    """Splits raw activations into train/val and normalizes by (mean, global std).

    NOTE: act_scale is a single global scalar (raw_acts.std() over all elements),
    not per-dimension. This matches the normalization scheme used throughout the
    project; a per-dimension normalization check was run separately and found
    NOT to explain the layer-quality differences observed (see project notes --
    layers with more uneven per-dimension variance were NOT consistently the
    harder-to-fit layers).
    """
    torch.manual_seed(seed)
    n = raw_acts.shape[0]
    n_train = int(n * train_frac)
    perm = torch.randperm(n)

    train_acts = raw_acts[perm[:n_train]]
    val_acts = raw_acts[perm[n_train:]]

    act_mean = train_acts.mean(dim=0).float()
    act_scale = train_acts.std().float()

    train_norm = ((train_acts - act_mean) / act_scale).float()
    val_norm = ((val_acts - act_mean) / act_scale).float()

    return {
        "train_norm": train_norm,
        "val_norm": val_norm,
        "act_mean": act_mean,
        "act_scale": act_scale,
    }
