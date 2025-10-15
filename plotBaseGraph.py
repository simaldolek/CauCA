import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Load adjacency (W) and lag (L)
W_hc = np.load("generatedSyntheticData/healthy_controls/base_graph/W_base.npy")
L_hc = np.load("generatedSyntheticData/healthy_controls/base_graph/L_base.npy")
W_ptsd = np.load("generatedSyntheticData/ptsd/base_graph/W_base.npy")
L_ptsd = np.load("generatedSyntheticData/ptsd/base_graph/L_base.npy")

# Build directed graphs
def build_graph(W, L):
    G = nx.DiGraph()
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if W[i, j] != 0:
                G.add_edge(j, i, lag=L[i, j])
    return G

G_hc = build_graph(W_hc, L_hc)
G_ptsd = build_graph(W_ptsd, L_ptsd)

# Layout
def get_pos(G):
    layers = list(nx.topological_generations(G))
    pos = {}
    for layer_idx, layer_nodes in enumerate(layers):
        y_positions = np.linspace(0, 1, len(layer_nodes))
        for i, node in enumerate(layer_nodes):
            pos[node] = (layer_idx, y_positions[i])
    return pos

pos_hc = get_pos(G_hc)
pos_ptsd = get_pos(G_ptsd)

# Gather all edge weights for consistent color mapping
all_weights = (
    [W_hc[dst, src] for src, dst in G_hc.edges()] +
    [W_ptsd[dst, src] for src, dst in G_ptsd.edges()]
)
vmin, vmax = min(all_weights), max(all_weights)
cmap = plt.cm.viridis

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Healthy Controls
nx.draw_networkx_nodes(G_hc, pos_hc, node_size=60, node_color="darkslategray", ax=axes[0])
nx.draw_networkx_edges(
    G_hc, pos_hc, ax=axes[0], arrows=True, arrowstyle="-|>", arrowsize=8, width=0.8,
    edge_color=[W_hc[dst, src] for src, dst in G_hc.edges()],
    edge_cmap=cmap, edge_vmin=vmin, edge_vmax=vmax
)
axes[0].set_title("Healthy Controls - Base Causal Graph", fontsize=14)
axes[0].axis("off")

# PTSD
nx.draw_networkx_nodes(G_ptsd, pos_ptsd, node_size=60, node_color="darkslategray", ax=axes[1])
edges_ptsd = nx.draw_networkx_edges(
    G_ptsd, pos_ptsd, ax=axes[1], arrows=True, arrowstyle="-|>", arrowsize=8, width=0.8,
    edge_color=[W_ptsd[dst, src] for src, dst in G_ptsd.edges()],
    edge_cmap=cmap, edge_vmin=vmin, edge_vmax=vmax
)
axes[1].set_title("PTSD - Base Causal Graph", fontsize=14)
axes[1].axis("off")

# Colorbar for edge weights (legend) next to PTSD plot
import matplotlib as mpl
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
fig.colorbar(sm, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04, label="Edge Weight")

plt.tight_layout()
plt.show()

print("Is DAG (Healthy)?", nx.is_directed_acyclic_graph(G_hc))
print("Is DAG (PTSD)?", nx.is_directed_acyclic_graph(G_ptsd))



