import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Load adjacency (W) and lag (L)
W_hc = np.load("generatedSyntheticData/healthy_controls/base_graph/W_base.npy")
L_hc = np.load("generatedSyntheticData/healthy_controls/base_graph/L_base.npy")

W_ptsd = np.load("generatedSyntheticData/ptsd/base_graph/W_base.npy")
L_ptsd = np.load("generatedSyntheticData/ptsd/base_graph/L_base.npy")

# Build directed graph
G_hc = nx.DiGraph()
for i in range(W_hc.shape[0]):
    for j in range(W_hc.shape[1]):
        if W_hc[i, j] != 0:
            G_hc.add_edge(j, i, lag=L_hc[i, j])

G_ptsd = nx.DiGraph()
for i in range(W_ptsd.shape[0]):
    for j in range(W_ptsd.shape[1]):
        if W_ptsd[i, j] != 0:
            G_ptsd.add_edge(j, i, lag=L_ptsd[i, j])

# --- Layout using topological ordering ---
# Get list of generations (layers) from source to sink
layers_hc = list(nx.topological_generations(G_hc))
layers_ptsd = list(nx.topological_generations(G_ptsd))

# Assign positions: x = layer index, y = evenly spaced within layer
pos_hc = {}
for layer_idx, layer_nodes in enumerate(layers_hc):
    n_nodes = len(layer_nodes)
    y_positions = np.linspace(0, 1, n_nodes)  # vertical spacing within layer
    for i, node in enumerate(layer_nodes):
        pos_hc[node] = (layer_idx, y_positions[i])

pos_ptsd = {}
for layer_idx, layer_nodes in enumerate(layers_ptsd):
    n_nodes = len(layer_nodes)
    y_positions = np.linspace(0, 1, n_nodes)  # vertical spacing within layer
    for i, node in enumerate(layer_nodes):
        pos_ptsd[node] = (layer_idx, y_positions[i])

# --- Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Healthy Controls
nx.draw_networkx_nodes(G_hc, pos_hc, node_size=60, node_color="darkslategray", ax=axes[0])
nx.draw_networkx_edges(G_hc, pos_hc, ax=axes[0], arrows=True, arrowstyle="-|>", arrowsize=8, width=0.8,
    edge_color=[W_hc[dst, src] for src, dst in G_hc.edges()])
axes[0].set_title("Healthy Controls - Base Causal Graph", fontsize=14)
axes[0].axis("off")

# PTSD
nx.draw_networkx_nodes(G_ptsd, pos_ptsd, node_size=60, node_color="darkslategray", ax=axes[1])
nx.draw_networkx_edges(G_ptsd, pos_ptsd, ax=axes[1], arrows=True, arrowstyle="-|>", arrowsize=8, width=0.8,
    edge_color=[W_ptsd[dst, src] for src, dst in G_ptsd.edges()])
axes[1].set_title("PTSD - Base Causal Graph", fontsize=14)
axes[1].axis("off")

plt.tight_layout()
plt.show()

# Check DAG status
print("Is DAG (Healthy)?", nx.is_directed_acyclic_graph(G_hc))
print("Is DAG (PTSD)?", nx.is_directed_acyclic_graph(G_ptsd))



