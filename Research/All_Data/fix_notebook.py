import json

with open("All_Data/Colab_SAE_Data_Analysis.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        sources = cell["source"]
        for i in range(len(sources)):
            if "!GIT_LFS_SKIP_SMUDGE" in sources[i]:
                sources[i] = "    pass # Using local paths instead of huggingface\n"
            if "if not os.path.exists(\"sae-rl-qwen05b-layers\"):" in sources[i]:
                sources[i] = "if False:\n"
            if "base_path = \"sae-rl-qwen05b-layers\"" in sources[i]:
                sources[i] = "base_path = \"../Michael/sae-rl-qwen05b-layers\"\n"
            
with open("All_Data/Colab_SAE_Data_Analysis.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

