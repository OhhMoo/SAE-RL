from google.colab import drive
drive.mount('/content/drive')

===

import os

drive_path = '/content/drive/MyDrive/SAE_RL_Research'
os.makedirs(drive_path, exist_ok=True)
%cd {drive_path}

if not os.path.exists("sae-rl-qwen05b-layers"):
    !GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/OhhMoo/sae-rl-qwen05b-layers

%cd sae-rl-qwen05b-layers
base_path = "."

===

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.optimize import linear_sum_assignment

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

chains = {
    "sae_flexible": "flexible",
    "sae_strict/k64": "strict_k64",
    "sae_strict/l23_k256": "strict_l23_k256",
    "sae_kl0p025": "kl0p025"
}

all_results = []
for rel_path, chain_label in chains.items():
    chain_path = os.path.join(base_path, rel_path)
    if not os.path.exists(chain_path): continue
        
    pt_files = glob.glob(os.path.join(chain_path, "*.pt"))
    for pt_file in pt_files:
        file_name = os.path.basename(pt_file)
        ckpt = torch.load(pt_file, map_location="cpu", weights_only=False)
        
        row = {'chain': chain_label, 'file_name': file_name, 'file_path': pt_file}
        layer_str = file_name.split("_layer")[1].split(".")[0]
        row['layer'] = f"layer{layer_str}"
        
        if "step" in file_name:
            try:
                row['step'] = int(file_name.split("step")[1].split("_")[0])
                row['model_type'] = "ppo"
            except: row['step'] = None
        else:
            row['step'] = 0
            row['model_type'] = "base"
            
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        for key, w_name in [('encoder_weight_norm', ['encoder.weight', 'W_enc']),
                            ('decoder_weight_norm', ['decoder.weight', 'W_dec'])]:
            row[key] = 0.0
            for name in w_name:
                if name in state_dict:
                    row[key] = state_dict[name].norm().item()
                    break
        all_results.append(row)

combined_df = pd.DataFrame(all_results)
if not combined_df.empty:
    combined_df = combined_df.sort_values(by=['chain', 'layer', 'step']).reset_index(drop=True)

===

sns.set_style("darkgrid")
base_models = combined_df[combined_df['model_type'] == 'base'].drop_duplicates(['chain', 'layer']).set_index(['chain', 'layer'])
ppo_models = combined_df[combined_df['model_type'] == 'ppo'].copy()

def get_base_norm(row):
    idx = (row['chain'], row['layer'])
    if idx in base_models.index: return base_models.loc[idx, 'encoder_weight_norm']
    fallback_idx = ('flexible', row['layer'])
    return base_models.loc[fallback_idx, 'encoder_weight_norm'] if fallback_idx in base_models.index else None

ppo_models['base_encoder_norm'] = ppo_models.apply(get_base_norm, axis=1)
ppo_models['encoder_norm_diff'] = ppo_models['encoder_weight_norm'] - ppo_models['base_encoder_norm']

g = sns.relplot(data=ppo_models, x="step", y="encoder_norm_diff", hue="chain", col="layer", col_wrap=2, kind="line", marker="o", height=4)
g.fig.suptitle("Encoder Norm Drift Over Time Across Configurations", y=1.05)
plt.show()

===

def plot_weight_norm_distributions_chains(layer_folder):
    layer_df = combined_df[combined_df['layer'] == layer_folder].sort_values('step')
    if layer_df.empty: return
    
    unique_chains = layer_df['chain'].unique()
    fig, axes = plt.subplots(1, len(unique_chains), figsize=(6*len(unique_chains), 5), sharex=True, sharey=True)
    if len(unique_chains) == 1: axes = [axes]
        
    for ax, chain_name in zip(axes, unique_chains):
        df_sub = layer_df[layer_df['chain'] == chain_name]
        colors = sns.color_palette("viridis", n_colors=len(df_sub))
        
        for i, (_, row) in enumerate(df_sub.iterrows()):
            ckpt = torch.load(row['file_path'], map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt)
            w_dec = state_dict.get("W_dec", state_dict.get("decoder.weight", next(iter(state_dict.values()))))
            if "decoder.weight" in state_dict: w_dec = w_dec.T
            
            norms = torch.norm(w_dec.float(), p=2, dim=-1)
            sns.kdeplot(norms.numpy(), color=colors[i], label=f"Step {row['step']}", alpha=0.7, ax=ax)
            
        ax.set_title(f"{chain_name} ({layer_folder})")
        ax.set_xlabel("L2 Norm")
        ax.legend()
        
    plt.suptitle(f"Decoder Feature Norm Distribution Over PPO Steps ({layer_folder})")
    plt.tight_layout()
    plt.show()

plot_weight_norm_distributions_chains("layer18")

===

def plot_mutual_orthogonality_chains(layer_folder):
    layer_df = combined_df[combined_df['layer'] == layer_folder].sort_values('step')
    if layer_df.empty: return
    
    ortho_results = []
    
    for i, row in layer_df.iterrows():
        ckpt = torch.load(row['file_path'], map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        w_dec = state_dict.get("W_dec", state_dict.get("decoder.weight", next(iter(state_dict.values()))))
        if "decoder.weight" in state_dict: w_dec = w_dec.T
            
        w_dec = w_dec.float()
        w_norm = w_dec / torch.norm(w_dec, p=2, dim=-1, keepdim=True)
        
        num_samples = min(2000, w_norm.shape[0])
        indices = torch.randperm(w_norm.shape[0])[:num_samples]
        w_sample = w_norm[indices]
        
        sim_matrix = torch.matmul(w_sample, w_sample.T)
        sim_matrix.fill_diagonal_(0)
        mean_ortho = torch.mean(torch.abs(sim_matrix)).item()
        
        ortho_results.append({'chain': row['chain'], 'step': row['step'], 'mean_ortho': mean_ortho})

    ortho_df = pd.DataFrame(ortho_results)
    
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=ortho_df, x="step", y="mean_ortho", hue="chain", marker='o')
    plt.title(f"Decoder Mean Mutual Feature Alignment in {layer_folder}")
    plt.ylabel("Mean |Cosine Similarity| (Off-Diagonal)")
    plt.grid(True)
    plt.show()

plot_mutual_orthogonality_chains("layer18")

===

BIRTH_DEATH_THRESHOLD = 0.7  

def get_normalized_decoder_features(state_dict):
    w_dec = state_dict.get('decoder.weight', state_dict.get('W_dec'))
    if w_dec is None: return None
    return torch.nn.functional.normalize(w_dec.float(), p=2, dim=0).cpu().numpy()

def match_features_hungarian(sim_matrix, threshold):
    cost_matrix = 1.0 - np.abs(sim_matrix) 
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    good_matches = np.abs(sim_matrix[row_ind, col_ind]) >= threshold
    return {
        "n_matched": int(good_matches.sum()),
        "n_births": len(set(range(sim_matrix.shape[1])) - set(col_ind[good_matches])),
        "n_deaths": len(set(range(sim_matrix.shape[0])) - set(row_ind[good_matches]))
    }

birth_death_results = []
target_layer = "layer18"

for chain in combined_df['chain'].unique():
    layer_files = combined_df[(combined_df['layer'] == target_layer) & (combined_df['chain'] == chain)].sort_values(by='step')
    if len(layer_files) <= 1: continue

    prev_ckpt = torch.load(layer_files.iloc[0]['file_path'], map_location="cpu", weights_only=False)
    W_dec_a = get_normalized_decoder_features(prev_ckpt.get('state_dict', prev_ckpt))
    
    for i in range(1, len(layer_files)):
        curr_row = layer_files.iloc[i]
        curr_ckpt = torch.load(curr_row['file_path'], map_location="cpu", weights_only=False)
        W_dec_b = get_normalized_decoder_features(curr_ckpt.get('state_dict', curr_ckpt))
        
        if W_dec_a is not None and W_dec_b is not None:
            stats = match_features_hungarian(W_dec_a.T @ W_dec_b, BIRTH_DEATH_THRESHOLD)
            birth_death_results.append({"chain": chain, "step_to": curr_row['step'], **stats})
        W_dec_a = W_dec_b

if birth_death_results:
    bd_df = pd.DataFrame(birth_death_results)
    chains_to_plot = [c for c in ["flexible", "strict_k64"] if c in bd_df['chain'].unique()]
    
    fig, axes = plt.subplots(1, len(chains_to_plot), figsize=(14, 5), sharey=True)
    if len(chains_to_plot) == 1: axes = [axes]
        
    for ax, chain_name in zip(axes, chains_to_plot):
        df_sub = bd_df[bd_df['chain'] == chain_name]
        steps = df_sub["step_to"].values
        ax.plot(steps, df_sub["n_births"], "o-", label="Births", color="green")
        ax.plot(steps, df_sub["n_deaths"], "s-", label="Deaths", color="red")
        ax.plot(steps, df_sub["n_matched"], "^-", label="Persisted", color="blue")
        ax.set_title(f"Feature Lifecycles ({target_layer}) - {chain_name}")
        ax.set_xlabel("RL Step")
        ax.legend()
    plt.tight_layout()
    plt.show()
