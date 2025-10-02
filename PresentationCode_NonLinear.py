import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import Trainer
from model.cauca_model import NonlinearCauCAModel
from data_generator.mixing_function import NonlinearMixing
from model.encoder import NonlinearCauCAEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pathlib import Path
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment

# Define a save directory
save_path = Path("mixing_details/")
save_path.mkdir(parents=True, exist_ok=True)

np.random.seed(26)
torch.manual_seed(0)

# Simple causal graph Z1 → Z2 → Z3
G = {
    1: [],      # Z1 has no parents
    2: [1],     # Z2 depends on Z1
    3: [2]      # Z3 depends on Z2
}


def sample_Z_obs(n):
    Z1 = np.random.normal(0, 1, n)
    Z2 = 0.5 * Z1 + np.random.normal(0, 0.5, n)
    Z3 = -0.3 * Z2 + np.random.normal(0, 0.5, n)
    return np.stack([Z1, Z2, Z3], axis=1)


def sample_Z_intervene_on_Z1(n):
    Z1 = np.random.choice([-2, 2], size=n) + np.random.normal(0, 0.3, n)
    Z2 = 0.5 * Z1 + np.random.normal(0, 0.5, n)
    Z3 = -0.3 * Z2 + np.random.normal(0, 0.5, n)
    
    return np.stack([Z1, Z2, Z3], axis=1)


def sample_Z_intervene_on_Z2(n):
    Z1 = np.random.normal(0, 1, n)
    Z2 = np.random.uniform(-2, 2, n)  
    Z3 = -0.3 * Z2 + np.random.normal(0, 0.5, n)
    return np.stack([Z1, Z2, Z3], axis=1)


def sample_Z_intervene_on_Z3(n):
    Z1 = np.random.normal(0, 1, n)
    Z2 = 0.5 * Z1 + np.random.normal(0, 0.5, n)
    Z3 = np.random.normal(2, 0.3, n)  
    return np.stack([Z1, Z2, Z3], axis=1)

# Create a mixing function with 3 layers (can change to 1, 2, etc.)
mixing_fn = NonlinearMixing(latent_dim=3, observation_dim=3, n_nonlinearities=3)



#### CREATE X AND Z ####


n_samples = 1000

# Environment 0 (observational)
Z_obs = sample_Z_obs(n_samples)
X_obs = mixing_fn(torch.tensor(Z_obs, dtype=torch.float32)).numpy()

# Environment 1 (intervention on Z1)
Z_int1 = sample_Z_intervene_on_Z1(n_samples)
X_int1 = mixing_fn(torch.tensor(Z_int1, dtype=torch.float32)).numpy()


# Environment 2 (intervention on Z2)
Z_int2 = sample_Z_intervene_on_Z2(n_samples)
X_int2 = mixing_fn(torch.tensor(Z_int2, dtype=torch.float32)).numpy()


# Environment 3 (intervention on Z3)
Z_int3 = sample_Z_intervene_on_Z3(n_samples)
X_int3 = mixing_fn(torch.tensor(Z_int3, dtype=torch.float32)).numpy()

datasets = {
    0: {'tau': [],     'X': X_obs,  'Z': Z_obs},   # Observational
    1: {'tau': [1],    'X': X_int1, 'Z': Z_int1},  # Intervene on Z1
    2: {'tau': [2],    'X': X_int2, 'Z': Z_int2},  # Intervene on Z2
    3: {'tau': [3],    'X': X_int3, 'Z': Z_int3},  # Intervene on Z3
}




### PLOT Z OF THE 4 ENVs ###

# Color map and labels
colors = {0: 'black', 1: 'red', 2: 'blue', 3: 'green'}
labels = {0: 'Observational', 1: 'Intervene Z1', 2: 'Intervene Z2', 3: 'Intervene Z3'}

# Concatenate all Z values for global min/max
all_Z = np.concatenate([datasets[env]['Z'] for env in datasets])

# Get global x-axis and y-axis limits for each Z variable
x_lims = []
y_max = []

# Compute limits for each Z dimension (Z1, Z2, Z3)
for j in range(3):
    z_vals = all_Z[:, j]
    x_min, x_max = np.min(z_vals), np.max(z_vals)
    x_lims.append((x_min, x_max))

    # Compute maximum histogram height across all environments
    y_vals = []
    for env_id in datasets:
        hist, _ = np.histogram(datasets[env_id]['Z'][:, j], bins=50, range=(x_min, x_max))
        y_vals.append(np.max(hist))
    y_max.append(max(y_vals))

# Create 4x3 subplot grid
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 12))

for env_id in range(4):
    Z = datasets[env_id]['Z']
    
    for j in range(3):  # Z1, Z2, Z3
        ax = axes[env_id, j]
        ax.hist(Z[:, j], bins=50, alpha=0.7, color=colors[env_id], range=x_lims[j])
        ax.set_xlim(x_lims[j])
        ax.set_ylim(0, y_max[j] * 1.1)  # Add a little padding to y-axis
        ax.set_xlabel(f"Z{j+1} value")
        

plt.tight_layout()
plt.show()






### PLOT X OF THE 4 ENVs ###

# Concatenate all X values for global min/max
all_X = np.concatenate([datasets[env]['X'] for env in datasets])

# Get global x-axis and y-axis limits for each X variable
x_lims = []
y_max = []

# Compute limits for each X dimension (X1, X2, X3)
for j in range(3):
    X_vals = all_X[:, j]
    x_min, x_max = np.min(X_vals), np.max(X_vals)
    x_lims.append((x_min, x_max))

    # Compute maximum histogram height across all environments
    y_vals = []
    for env_id in datasets:
        hist, _ = np.histogram(datasets[env_id]['X'][:, j], bins=50, range=(x_min, x_max))
        y_vals.append(np.max(hist))
    y_max.append(max(y_vals))

# Create 4x3 subplot grid
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 12))

for env_id in range(4):
    X = datasets[env_id]['X']
    
    for j in range(3):  # X1, X2, X3
        ax = axes[env_id, j]
        ax.hist(X[:, j], bins=50, alpha=0.7, color=colors[env_id], range=x_lims[j])
        ax.set_xlim(x_lims[j])
        ax.set_ylim(0, y_max[j] * 1.1)
        ax.set_xlabel(f"X{j+1} value")
        
        # Add column titles for top row
        if env_id == 0:
            ax.set_title(f"X{j+1}")


plt.tight_layout()
plt.show()


mixing_fn.save_coeffs(save_path)
matrix_0 = pd.read_csv("mixing_details/matrix_0.csv")
print("Matrix 0:")
print(matrix_0)

# Number of layers
print("Number of nonlinear layers:", mixing_fn.n_nonlinearities)

# List of invertible matrices used
for i, mat in enumerate(mixing_fn.matrices):
    print(f"\nMatrix {i}:\n", mat)

# Print determinant of each matrix
for i, mat in enumerate(mixing_fn.matrices):
    det = torch.det(mat)
    print(f"Determinant of matrix {i}: {det.item():.4f}")



### RUN CAUCA ###

dim = 3  # since Z and X are both 3-dimensional

# Define the adjacency matrix for Z1 → Z2 → Z3
your_adjacency_matrix = np.array([
    [0, 1, 0],  # Z1 → Z2
    [0, 0, 1],  # Z2 → Z3
    [0, 0, 0]
])

# Define intervention targets per environment as a tensor
intervention_targets_tensor = torch.tensor([
    [0, 0, 0],  # observational
    [1, 0, 0],  # intervention on Z1
    [0, 1, 0],  # intervention on Z2
    [0, 0, 1],  # intervention on Z3
], dtype=torch.int)



epochs = 5000
early_stopping_patience = 300  # Stop if no improvement in 300 epochs


# Stack all environments' data
x = torch.tensor(np.concatenate([datasets[k]['X'] for k in range(4)]), dtype=torch.float32)

# Environment label per sample (0,1,2,3)
e = torch.cat([
    torch.full((n_samples,), k, dtype=torch.int) for k in range(4)
])

# Intervention targets per sample (repeat each env's intervention mask n_samples times)
intervention_targets = torch.cat([
    intervention_targets_tensor[k].repeat(n_samples, 1) for k in range(4)
], dim=0)


def train_and_evaluate(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = NonlinearCauCAEncoder(
        latent_dim=dim,
        adjacency_matrix=your_adjacency_matrix,
        intervention_targets_per_env=intervention_targets_tensor,
        K=3,
        net_hidden_dim=128,
        net_hidden_layers=3
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_val = float('-inf')
    patience = 0

    model.train()
    for epoch in range(epochs):
        log_probs, _ = model.multi_env_log_prob(x, e, intervention_targets)
        loss = -log_probs.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        current_val = -loss.item()  # same as log_probs.mean().item()
        if current_val > best_val:
            best_val = current_val
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} with best validation log-prob: {best_val:.4f}")
                break

    model.eval()
    with torch.no_grad():
        log_prob_val, _ = model.multi_env_log_prob(x, e, intervention_targets)
        val_log_prob = log_prob_val.mean().item()

    return model, val_log_prob



all_models = []
for seed in [0, 1, 2]:
    model_i, val_log_prob_i = train_and_evaluate(seed)
    all_models.append((model_i, val_log_prob_i))

# Choose best model
best_model, best_val_log_prob = max(all_models, key=lambda x: x[1])
print(f"\nSelected best model with validation log-prob: {best_val_log_prob:.4f}")

Z_hat = best_model(x)

# Ground truth Z: concatenate across all 4 environments
Z_true = np.concatenate([datasets[k]['Z'] for k in range(4)], axis=0)

# Estimated Z from the trained model
Z_hat_np = Z_hat.detach().numpy()




# ----- GLOBAL MEAN CORRELATION COEFFICIENT (MCC) -----

# Compute correlation matrix between true and estimated Zs
correlation_matrix = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        correlation_matrix[i, j] = abs(pearsonr(Z_true[:, i], Z_hat_np[:, j])[0])

# Use Hungarian algorithm to find best matching (maximize total correlation)
row_ind, col_ind = linear_sum_assignment(-correlation_matrix)
mcc = correlation_matrix[row_ind, col_ind].mean()

print(f"\nMean Correlation Coefficient (MCC): {mcc:.4f}")
print("Correlation matrix:\n", correlation_matrix)
print("Best matching (true ↔ learned):", list(zip(row_ind, col_ind)))



# ----- PER ENV MEAN CORRELATION COEFFICIENT (MCC) -----

per_env_mcc = []

for k in range(4):
    X_k = torch.tensor(datasets[k]['X'], dtype=torch.float32)
    with torch.no_grad():
        Z_hat_k = best_model(X_k).cpu().numpy()
    Z_true_k = datasets[k]['Z']  # (n_samples, 3) numpy

    corr_k = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            r = pearsonr(Z_true_k[:, i], Z_hat_k[:, j])[0]
            corr_k[i, j] = np.abs(np.nan_to_num(r, nan=0.0))

    ri, cj = linear_sum_assignment(-corr_k)
    mcc_k = corr_k[ri, cj].mean()
    per_env_mcc.append(mcc_k)

for k, m in enumerate(per_env_mcc):
    print(f"Env {k}: MCC = {m:.4f}")





best_model.eval()

# Storage for plotting
Z_hat_envs = {}

# Recover Z_hat per environment
for k in range(4):
    X_k = torch.tensor(datasets[k]['X'], dtype=torch.float32)
    with torch.no_grad():
        Z_hat_k = best_model(X_k).cpu().numpy()
    Z_hat_envs[k] = Z_hat_k

# Color map for environments
colors = {0: 'black', 1: 'red', 2: 'blue', 3: 'green'}
labels = {0: 'Observational', 1: 'Intervene Z1', 2: 'Intervene Z2', 3: 'Intervene Z3'}

# Get global min/max for each Z_hat dim
all_Z_hat = np.vstack(list(Z_hat_envs.values()))
x_lims = [(all_Z_hat[:, j].min(), all_Z_hat[:, j].max()) for j in range(3)]

# Create figure
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 10))

for env_id in range(4):
    Z_hat_k = Z_hat_envs[env_id]
    for j in range(3):
        ax = axes[env_id, j]
        ax.hist(Z_hat_k[:, j], bins=50, alpha=0.7, color=colors[env_id], range=x_lims[j])
        ax.set_xlim(x_lims[j])
        ax.set_xlabel(f"Ẑ{j+1} value")
        if j == 0:
            ax.set_ylabel(labels[env_id])
        if env_id == 0:
            ax.set_title(f"Ẑ{j+1}")

plt.tight_layout()
plt.show()



