import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import numpy as np
import pandas as pd
from data_generator.multi_env_gdp import make_multi_env_dgp

latent_dim = 7 # The number of unobserved causal variables
observation_dim = 170  # AAL3 The number of measured variables —what the model "observes" (BOLD time series). 


# Option 1: amygdala, PFC, hippocampus, ACC, insula, thalamus, DMN - (need exact ROI mapping)
# Option 2: DMN, salience, visual, sensorimotor, FPN, limbic, subcortical - (less interpretable, e.g., how do we exactly intervene on DMN?)

# Define 4 environments: 0 = control, 1 = PTSD_light, 2 = PTSD_mod, 3 = PTSD_severe
intervention_targets_per_env = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0],  # Control
    [1, 0, 0, 0, 0, 0, 0],  # PTSD_Light: Amygdala
    [1, 0, 1, 0, 0, 0, 0],  # PTSD_Moderate: Amygdala + Hippocampus
    [1, 1, 1, 1, 0, 0, 0],  # PTSD_Severe: + PFC, ACC
]).int()

# Optional: scale noise per environment to reflect greater variability in PTSD
env_noise_scale = torch.tensor([  # 1.0 = baseline (control), >1 = noisier dynamics
    1.0,   # control
    1.2,   # PTSD_light
    1.5,   # PTSD_moderate
    2.0    # PTSD_severe
])

# DAG Directionality - based on lit review
# Order: [Amygdala, PFC, Hippocampus, ACC, Insula, Thalamus, DMN]
adjacency_matrix = np.array([
    [0,  0,  0,  0,  0,  0,  0],  # Amygdala
    [1,  0,  1,  0,  0,  0,  1],  # PFC (→ Amygdala, Hippocampus, DMN)
    [0,  0,  0,  0,  0,  0,  0],  # Hippocampus
    [1,  0,  0,  0,  1,  0,  0],  # ACC (→ Amygdala, Insula)
    [0,  0,  0,  0,  0,  0,  1],  # Insula (→ DMN)
    [1,  0,  1,  0,  0,  0,  0],  # Thalamus (→ Amygdala, Hippocampus)
    [0,  0,  0,  0,  0,  0,  0],  # DMN
])

manual_latent_map = {
    "Amygdala": [45, 46],
    "PFC": [3, 4, 5, 6, 21, 22],
    "Hippocampus": [41, 42],
    "ACC": [151, 152, 153, 154, 155, 156],
    "Insula": [33, 34],
    "Thalamus": [121, 122, 125, 126, 127, 128, 129, 130],
    "DMN": [19, 20, 20, 39, 40, 69, 71, 72]  # duplicate '20' is intentional
}

latent_regions_ordered = ["Amygdala", "PFC", "Hippocampus", "ACC", "Insula", "Thalamus", "DMN"]

#Rows = each of the 170 observed ROIs (AAL3)
#Columns = each of the 7 latent brain regions
manual_mixing_matrix = np.zeros((170, len(latent_regions_ordered)))

for i, latent in enumerate(latent_regions_ordered):
    roi_indices = list(set(manual_latent_map[latent]))  # remove duplicates
    for roi_idx in roi_indices:
        if roi_idx < 170:
            manual_mixing_matrix[roi_idx, i] = 1.0

# summarize the matrix
manual_counts = np.sum(manual_mixing_matrix, axis=0).astype(int)

manual_roi_df = pd.DataFrame({
    "Latent Region": latent_regions_ordered,
    "ROI Count": manual_counts
})

print(manual_roi_df)

from data_generator.multi_env_gdp import make_multi_env_dgp

dgp = make_multi_env_dgp(
    latent_dim=latent_dim,
    observation_dim=observation_dim,
    adjacency_matrix=adjacency_matrix,  # Random DAG if not specified
    intervention_targets_per_env=intervention_targets_per_env,
    mixing="nonlinear",     
    scm="location-scale",           # "location-scale" instead of "linear" for simulate PTSD-induced instability (determines how nodes interact with each other)
    shift_noise=True,
    noise_shift_type="both", # mean + variance noise
    env_noise_scale=env_noise_scale,
    mixing_matrix=manual_mixing_matrix,
)

x, v, u, e, targets, log_prob = dgp.sample(num_samples_per_env=200, intervention_targets_per_env=intervention_targets_per_env)

print("x shape:", x.shape)
print("First 5 rows of observed data (x):")
print(x[:5])

torch.save(x, "synthetic_x.pt")
torch.save(v, "synthetic_latents.pt")
torch.save(e, "synthetic_env_labels.pt")



import matplotlib.pyplot as plt
import seaborn as sns

# Move to CPU and convert to NumPy
x_np = x.cpu().numpy()
v_np = v.cpu().numpy()
e_np = e.cpu().numpy().flatten()

# Create a DataFrame for observed data with environment labels
df = pd.DataFrame(x_np)
df['env'] = e_np
df['env_label'] = df['env'].map({
    0: "Control",
    1: "PTSD_Light",
    2: "PTSD_Moderate",
    3: "PTSD_Severe"
})

# --- ✦ 1. Plot mean ROI activity per environment ---
mean_by_env = df.groupby("env_label").mean().T  # ROIs x Envs

plt.figure(figsize=(12, 6))
sns.heatmap(mean_by_env, cmap="coolwarm", center=0)
plt.title("Mean ROI Activity by Environment")
plt.xlabel("Environment")
plt.ylabel("ROI Index")
plt.tight_layout()
plt.savefig("mean_ROI_by_env.png", dpi=300)
plt.show()

# --- ✦ 2. Variance per environment ---
var_by_env = df.groupby("env_label").var().T

plt.figure(figsize=(12, 6))
sns.heatmap(var_by_env, cmap="magma", center=0)
plt.title("ROI Variance by Environment")
plt.xlabel("Environment")
plt.ylabel("ROI Index")
plt.tight_layout()
plt.savefig("variance_ROI_by_env.png", dpi=300)
plt.show()

# --- ✦ 3. PCA for dimensionality reduction and scatter plot ---
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_np)

plt.figure(figsize=(8, 6))
for i, label in enumerate(["Control", "PTSD_Light", "PTSD_Moderate", "PTSD_Severe"]):
    plt.scatter(x_pca[e_np == i, 0], x_pca[e_np == i, 1], label=label, alpha=0.6)

plt.title("PCA of Observed ROI Activity")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.savefig("pca_ROI_activity.png", dpi=300)
plt.show()

# --- ✦ 4. Save all plots and data summaries ---
mean_by_env.to_csv("mean_ROI_by_env.csv")
var_by_env.to_csv("variance_ROI_by_env.csv")

print("Visualizations saved as PNG files. Data summaries saved as CSV.")
