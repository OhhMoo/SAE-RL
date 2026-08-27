import json

with open("All_Data/Colab_SAE_Data_Analysis.ipynb", "r") as f:
    nb = json.load(f)

# Keep cells up to index 15 (which is right before "6. Deep Checkpoint Analysis")
new_cells = nb["cells"][:15]

csv_cells = [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Deep CSV Metrics Analysis Across Researchers\n",
    "Since we don't have the `.pt` weights for Jacob and Tyler, we can instead do a detailed dive into the CSV summary statistics across all three datasets."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "sns.set_theme(style=\"darkgrid\")\n",
    "\n",
    "plt.figure(figsize=(12, 6))\n",
    "\n",
    "if not df_michael.empty:\n",
    "    sns.scatterplot(data=df_michael, x=\"Avg L0\", y=\"Recon Loss (MSE)\", hue=\"Chain\", style=\"Layer\", s=100, palette=\"viridis\")\n",
    "    plt.title(\"Michael: MSE vs L0 Grouped by Chain and Layer\")\n",
    "    plt.xscale('log')\n",
    "    plt.yscale('log')\n",
    "    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if not df_tyler.empty:\n",
    "    fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
    "    \n",
    "    sns.scatterplot(data=df_tyler, x=\"hard_l0\", y=\"explained_var\", hue=\"expansion_factor\", ax=axes[0], s=80, palette=\"magma\")\n",
    "    axes[0].set_title(\"Tyler: Explained Variance vs Hard L0\")\n",
    "    \n",
    "    sns.scatterplot(data=df_tyler, x=\"hard_l0\", y=\"dead_frac_batch\", hue=\"expansion_factor\", ax=axes[1], s=80, palette=\"magma\")\n",
    "    axes[1].set_title(\"Tyler: Dead Fraction vs Hard L0\")\n",
    "    \n",
    "    sns.scatterplot(data=df_tyler, x=\"mean_threshold\", y=\"fire_rate_mean\", hue=\"expansion_factor\", ax=axes[2], s=80, palette=\"magma\")\n",
    "    axes[2].set_title(\"Tyler: Mean Fire Rate vs Threshold\")\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if not df_michael.empty:\n",
    "    plt.figure(figsize=(10, 5))\n",
    "    sns.boxplot(data=df_michael, x=\"Layer\", y=\"Dead Latents %\", hue=\"Chain\")\n",
    "    plt.title(\"Michael: Dead Latents Percentage Across Layers and Chains\")\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if not df_jacob.empty:\n",
    "    plt.figure(figsize=(10, 5))\n",
    "    sns.barplot(data=df_jacob, x=\"Layer\", y=\"Frac Rec %\", hue=\"K\", palette=\"Blues_d\")\n",
    "    plt.title(\"Jacob: Fractional Recovery % by Layer and K-value\")\n",
    "    plt.show()\n",
    "    \n",
    "    plt.figure(figsize=(10, 5))\n",
    "    sns.scatterplot(data=df_jacob, x=\"Avg L0\", y=\"Recon Loss (MSE)\", hue=\"Layer\", size=\"K\", sizes=(50, 200), palette=\"coolwarm\")\n",
    "    plt.title(\"Jacob: MSE vs L0 Tradeoff (Bubble size = K)\")\n",
    "    plt.xscale('log')\n",
    "    plt.yscale('log')\n",
    "    plt.show()"
   ]
  }
]

new_cells.extend(csv_cells)
new_cells.extend(nb["cells"][15:])

nb["cells"] = new_cells

with open("All_Data/Colab_SAE_Data_Analysis.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

