"""HuggingFace Hub upload helpers.

Requires an HF write token, read from the HF_WRITE_TOKEN environment variable
by default (do not hardcode tokens in scripts committed to the repo).
"""
import json
import os

import torch
from huggingface_hub import HfApi, create_repo


def get_api(token: str = None) -> HfApi:
    token = token or os.environ.get("HF_WRITE_TOKEN")
    if not token:
        raise ValueError(
            "No HF write token provided. Set the HF_WRITE_TOKEN environment "
            "variable or pass token= explicitly."
        )
    return HfApi(token=token)


def ensure_repo(repo_id: str, api: HfApi, repo_type: str = "model", private: bool = False):
    """Creates the repo if it doesn't already exist. Safe to call repeatedly."""
    try:
        create_repo(repo_id=repo_id, repo_type=repo_type, token=api.token, private=private)
    except Exception as e:
        # Repo likely already exists -- HfApi doesn't expose a clean "already exists"
        # check without an extra network call, so we log and continue rather than
        # silently swallowing genuine auth/permission errors.
        print(f"create_repo note (may already exist): {e}")


def save_and_upload_checkpoint(
    sae,
    repo_id: str,
    filename: str,
    api: HfApi,
    metadata: dict,
    local_dir: str = "uploads",
    cleanup_local: bool = True,
):
    """Saves an SAE's state_dict + metadata to disk, uploads it, then removes
    the local copy (cleanup_local=True) to avoid disk/memory buildup across
    a multi-checkpoint sweep.
    """
    os.makedirs(local_dir, exist_ok=True)
    ckpt = {"sae_state_dict": sae.state_dict(), **metadata}

    local_path = os.path.join(local_dir, filename)
    torch.save(ckpt, local_path)

    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"  Uploaded {filename}")

    if cleanup_local:
        os.remove(local_path)


def upload_json_summary(data: dict, repo_id: str, filename: str, api: HfApi, local_dir: str = "uploads"):
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)
    with open(local_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded {filename}")
