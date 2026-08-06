"""Upload the intermediate GSM8k-SFT checkpoints to the Hugging Face Hub.

Pushes each local checkpoint's huggingface/ dir to
OhhMoo/qwen05b-gsm8k-sft-instruct under checkpoints/global_step_<N>/,
skipping steps whose model.safetensors is already complete on the Hub.
Safe to re-run after an interruption.
"""

import os

from huggingface_hub import HfApi

REPO = "OhhMoo/qwen05b-gsm8k-sft-instruct"
CKPT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "checkpoints/gsm8k-sft/qwen05b-gsm8k-sft-instruct-ckpts",
)
STEPS = [29, 58, 87, 116, 145, 174, 203, 232, 261, 290, 319, 348]


def main():
    api = HfApi()
    done = {}
    paths = [f"checkpoints/global_step_{s}/model.safetensors" for s in STEPS]
    for info in api.get_paths_info(REPO, paths):
        done[info.path] = info.size

    for step in STEPS:
        local = os.path.join(CKPT_ROOT, f"global_step_{step}", "huggingface")
        remote = f"checkpoints/global_step_{step}"
        local_size = os.path.getsize(os.path.join(local, "model.safetensors"))
        if done.get(f"{remote}/model.safetensors") == local_size:
            print(f"step {step}: already on Hub, skipping")
            continue
        print(f"step {step}: uploading {local} -> {remote}")
        api.upload_folder(
            repo_id=REPO,
            folder_path=local,
            path_in_repo=remote,
            repo_type="model",
            commit_message=f"Add SFT checkpoint global_step_{step}",
        )
    print("ALL CHECKPOINTS UPLOADED: https://huggingface.co/" + REPO)


if __name__ == "__main__":
    main()
