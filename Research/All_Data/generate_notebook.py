import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# SAE and RL Training Data Analysis\n",
    "This notebook analyzes the three sets of SAE data metrics: Jacob's, Michael's, and Tyler's.\n",
    "Run this in Colab after uploading your `All_Data` folder."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "sns.set_theme(style=\"darkgrid\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Data Loading"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# In local mode, assuming we are running inside the folder or have the paths correct.\n",
    "# If on Colab, you might need to adjust the paths below.\n",
    "base_path = \".\"\n",
    "path_jacob = os.path.join(base_path, \"SAE_Collation - sae_collation_jacob.csv\")\n",
    "path_tyler = os.path.join(base_path, \"SAE_Collation - sae_collation_tyler.csv\")\n",
    "path_michael = os.path.join(base_path, \"SAE_Collation - sae_michael.csv\")\n",
    "\n",
    "df_jacob = pd.read_csv(path_jacob) if os.path.exists(path_jacob) else pd.DataFrame()\n",
    "df_tyler = pd.read_csv(path_tyler) if os.path.exists(path_tyler) else pd.DataFrame()\n",
    "df_michael = pd.read_csv(path_michael) if os.path.exists(path_michael) else pd.DataFrame()\n",
    "\n",
    "print(f\"Jacob's Data: {df_jacob.shape}\")\n",
    "print(f\"Tyler's Data: {df_tyler.shape}\")\n",
    "print(f\"Michael's Data: {df_michael.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Unifying Data (Pareto Metrics)\n",
    "We want to look at L0 (or approximate sparsity) vs MSE / Recon Loss."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_data = []\n",
    "\n",
    "if not df_jacob.empty:\n",
    "    df_j = df_jacob.copy()\n",
    "    df_j['Source'] = 'Jacob'\n",
    "    df_j['L0'] = df_j['Avg L0']\n",
    "    df_j['MSE'] = df_j['Recon Loss (MSE)']\n",
    "    df_j['Dead_Latent_Frac'] = df_j['Dead Latents %'] / 100.0\n",
    "    plot_data.append(df_j[['Source', 'L0', 'MSE', 'Expansion', 'Dead_Latent_Frac']])\n",
    "\n",
    "if not df_michael.empty:\n",
    "    df_m = df_michael.copy()\n",
    "    df_m['Source'] = 'Michael (chain: ' + df_m['Chain'].fillna('none') + ')'\n",
    "    df_m['L0'] = df_m['Avg L0']\n",
    "    df_m['MSE'] = df_m['Recon Loss (MSE)']\n",
    "    df_m['Dead_Latent_Frac'] = df_m['Dead Latents %'] / 100.0\n",
    "    plot_data.append(df_m[['Source', 'L0', 'MSE', 'Expansion', 'Dead_Latent_Frac']])\n",
    "\n",
    "if not df_tyler.empty:\n",
    "    df_t = df_tyler.copy()\n",
    "    df_t['Source'] = 'Tyler'\n",
    "    df_t['L0'] = df_t['hard_l0']\n",
    "    df_t['MSE'] = df_t['mse']\n",
    "    df_t['Dead_Latent_Frac'] = df_t['dead_frac_batch']\n",
    "    # Convert expansion string to int if possible\n",
    "    if df_t['expansion_factor'].dtype == object:\n",
    "        df_t['Expansion'] = df_t['expansion_factor'].str.replace('x', '').astype(float)\n",
    "    else:\n",
    "        df_t['Expansion'] = df_t['expansion_factor']\n",
    "    plot_data.append(df_t[['Source', 'L0', 'MSE', 'Expansion', 'Dead_Latent_Frac']])\n",
    "\n",
    "if plot_data:\n",
    "    df_all = pd.concat(plot_data, ignore_index=True)\n",
    "    print(\"Unified DataFrame preview:\", df_all.head())\n",
    "else:\n",
    "    df_all = pd.DataFrame()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Analysis Plots"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if not df_all.empty:\n",
    "    plt.figure(figsize=(10, 6))\n",
    "    sns.scatterplot(data=df_all, x=\"L0\", y=\"MSE\", hue=\"Source\", style=\"Expansion\", s=100, alpha=0.8)\n",
    "    plt.title(\"Pareto Frontier: MSE vs L0\")\n",
    "    plt.xlabel(\"L0 (Average active latents)\")\n",
    "    plt.ylabel(\"Reconstruction MSE\")\n",
    "    plt.yscale('log')\n",
    "    plt.xscale('log')\n",
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
    "if not df_all.empty:\n",
    "    plt.figure(figsize=(10, 6))\n",
    "    sns.scatterplot(data=df_all, x=\"L0\", y=\"Dead_Latent_Frac\", hue=\"Source\", style=\"Expansion\", s=100, alpha=0.8)\n",
    "    plt.title(\"Dead Latents vs L0\")\n",
    "    plt.xlabel(\"L0 (Average active latents)\")\n",
    "    plt.ylabel(\"Dead Latent Fraction\")\n",
    "    plt.xscale('log')\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Deep Dive into Tyler's Detailed Firing Metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if not df_tyler.empty:\n",
    "    fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n",
    "    sns.scatterplot(data=df_tyler, x=\"features_gt_10pct_firing\", y=\"explained_var\", hue=\"expansion_factor\", ax=axes[0], s=80)\n",
    "    axes[0].set_title(\"Explained Var vs highly active features\")\n",
    "    \n",
    "    sns.lineplot(data=df_tyler.sort_values(\"hard_l0\"), x=\"hard_l0\", y=\"explained_var\", hue=\"expansion_factor\", marker=\"o\", ax=axes[1])\n",
    "    axes[1].set_title(\"Explained Variance Pareto\")\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Summary Statistics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if not df_jacob.empty:\n",
    "    print(\"\\nJacob Data By Layer:\")\n",
    "    display(df_jacob.groupby('Layer')[['Avg L0', 'Recon Loss (MSE)']].mean())\n",
    "    \n",
    "if not df_michael.empty:\n",
    "    print(\"\\nMichael Data By Chain:\")\n",
    "    display(df_michael.groupby('Chain')[['Avg L0', 'Recon Loss (MSE)']].mean())\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("All_Data/Colab_SAE_Data_Analysis.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

