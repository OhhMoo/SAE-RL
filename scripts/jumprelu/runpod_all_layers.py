"""Train, upload, verify, and clean every Qwen SAE layer on RunPod."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "tchalfpenny/qwen2.5-0.5b-gsm8k-jumprelu-saes"
WORK_DIR = Path("/workspace/qwen_sae")
DATASET_SLUG = "tchalfpenny--qwen2.5-0.5b-gsm8k-rollouts"
TRAINING_SCRIPT = Path("/workspace/sae_train_modular.py")
STATUS_PATH = WORK_DIR / "all_layers_status.tsv"
LAYERS = [layer for layer in range(24) if layer != 6]


def timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def record(layer: int, status: str) -> None:
    with STATUS_PATH.open("a") as handle:
        handle.write(f"{layer}\t{status}\t{timestamp()}\n")


def layer_dir(layer: int) -> Path:
    return WORK_DIR / "saes" / DATASET_SLUG / f"layer_{layer:02d}_sweep"


def activation_dir(layer: int) -> Path:
    return WORK_DIR / "activations" / DATASET_SLUG / f"layer_{layer:02d}"


def remote_layer_is_complete(api: HfApi, layer: int) -> bool:
    prefix = f"layers/layer_{layer:02d}/"
    files = api.list_repo_files(REPO_ID, repo_type="model")
    layer_files = [name for name in files if name.startswith(prefix)]
    checkpoints = [name for name in layer_files if name.endswith(".pt")]
    return (
        len(layer_files) == 129
        and len(checkpoints) == 64
        and prefix + "sweep_results.json" in files
    )


def verify_local_layer(layer: int) -> None:
    directory = layer_dir(layer)
    checkpoints = list(directory.glob("*.pt"))
    results = directory / "sweep_results.json"
    if len(checkpoints) != 64 or not results.exists():
        raise RuntimeError(
            f"Layer {layer} incomplete locally: "
            f"{len(checkpoints)} checkpoints, results={results.exists()}"
        )


def train_layer(layer: int) -> None:
    log_path = WORK_DIR / f"layer_{layer:02d}.log"
    environment = os.environ.copy()
    environment.update(
        {
            "QWEN_SAE_PILOT": "0",
            "QWEN_SAE_WORKERS": "8",
            "QWEN_SAE_LAYER": str(layer),
            "QWEN_SAE_WORK_DIR": str(WORK_DIR),
        }
    )
    for attempt in range(1, 4):
        with log_path.open("a") as log:
            log.write(f"\n=== controller attempt {attempt} at {timestamp()} ===\n")
            completed = subprocess.run(
                [sys.executable, "-u", str(TRAINING_SCRIPT)],
                cwd="/workspace",
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode == 0:
            return
        print(f"Layer {layer}: training attempt {attempt} failed", flush=True)
        time.sleep(15)
    raise RuntimeError(f"Layer {layer}: training failed three times; see {log_path}")


def upload_layer(layer: int) -> None:
    remote_path = f"layers/layer_{layer:02d}"
    for attempt in range(1, 4):
        completed = subprocess.run(
            [
                "hf",
                "upload",
                REPO_ID,
                str(layer_dir(layer)),
                remote_path,
                "--commit-message",
                f"Upload complete layer {layer}",
            ],
            check=False,
        )
        if completed.returncode == 0:
            return
        print(f"Layer {layer}: upload attempt {attempt} failed", flush=True)
        time.sleep(30)
    raise RuntimeError(f"Layer {layer}: upload failed three times")


def upload_status() -> None:
    subprocess.run(
        [
            "hf",
            "upload",
            REPO_ID,
            str(STATUS_PATH),
            "training/all_layers_status.tsv",
            "--commit-message",
            "Update layer training status",
        ],
        check=True,
    )


def main() -> None:
    api = HfApi()
    for layer in LAYERS:
        print(f"=== Layer {layer} starting at {timestamp()} ===", flush=True)
        if remote_layer_is_complete(api, layer):
            print(f"Layer {layer}: already verified remotely", flush=True)
        else:
            record(layer, "training")
            train_layer(layer)
            verify_local_layer(layer)
            record(layer, "trained")
            upload_layer(layer)
            if not remote_layer_is_complete(api, layer):
                raise RuntimeError(f"Layer {layer}: remote verification failed")
            record(layer, "uploaded_verified")

        # User authorized deletion only after successful remote verification.
        shutil.rmtree(layer_dir(layer), ignore_errors=True)
        shutil.rmtree(activation_dir(layer), ignore_errors=True)
        record(layer, "uploaded_verified_cleaned")
        upload_status()
        usage = shutil.disk_usage(WORK_DIR)
        print(
            f"Layer {layer}: complete; {usage.free / 2**30:.1f} GiB free",
            flush=True,
        )

    print("ALL_LAYERS_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
