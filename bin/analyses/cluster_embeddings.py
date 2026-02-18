# Read this file line by line
import torch
import numpy as np
from safetensors.torch import load_file
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Cluster Evo2 Embeddings")
    parser.add_argument('--input', type=str, required=True, help="Path to .safetensors file")
    parser.add_argument('--output', type=str, default="cluster_plot.png", help="Output image path")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Loading {args.input}...")
    data = load_file(args.input)

    vectors = []
    labels = []
    keys = []

    # 1. Process Data
    print("Pooling embeddings...")
    for key, tensor in data.items():
        # Shape is (1, 107, 4096)
        # We want to squash the middle dimension (107) to get a general 'sequence summary'
        # Result shape: (4096,)
        # so, 'squashed' it with mean - imo a dumb idea
        pooled_vector = torch.mean(tensor.squeeze(0), dim=0).float().numpy()

        vectors.append(pooled_vector)
        keys.append(key)

        # Extract Label (e.g., 'hom' from 'hom_44_1027084')
        label = key.split('_')[0] if '_' in key else 'Unknown'
        labels.append(label)

    X = np.array(vectors)
    print(f"Data Shape: {X.shape}") # Should be (Num_Sequences, 4096)

    # 2. PCA (Sanity Check & Denoising)
    # 4096 dimensions is too high for direct clustering/t-SNE.
    # We reduce to 50 components first to remove noise.
    print("Running PCA...")
    pca = PCA(n_components=min(50, len(X)))
    X_pca = pca.fit_transform(X)

    print(f"Explained Variance (First 2 Comps): {np.sum(pca.explained_variance_ratio_[:2]):.2%}")

    # 3. t-SNE (Visualization)
    print("Running t-SNE...")
    # Perplexity must be less than number of samples
    perp = min(30, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X_pca)

    # 4. K-Means Clustering (Optional: To see if math agrees with labels)
    # We assume k = number of unique labels found
    num_clusters = len(set(labels))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X_pca)

    # 5. Plotting
    print("Plotting...")
    plt.figure(figsize=(12, 5))

    # Subplot 1: Colored by True Label (hom/het/rep)
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=labels, palette="deep", s=60)
    plt.title("t-SNE: Colored by Label (hom/het)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    # Subplot 2: Colored by K-Means Cluster
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=cluster_ids, palette="viridis", s=60, legend="full")
    plt.title(f"t-SNE: K-Means Clustering (k={num_clusters})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
# /usr/bin/env python

import torch
from safetensors.torch import load_file

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--safetensor', type = str, required = True)
args = parser.parse_args()

# Load the file
file_path = args.safetensor
embeddings = load_file(file_path)

# Print keys and shapes
for key, tensor in embeddings.items():
    print(f"Key: {key}")
    print(f"Shape: {tensor.shape}")
import os
os.environ['LOKY_BACKEND'] = 'threading'

import torch
import numpy as np
from safetensors.torch import load_file
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Path to .safetensors")
    parser.add_argument('--output', type=str, default="umap_cosine.png")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Loading {args.input}...")
    data = load_file(args.input)

    vectors = []
    labels = []

    print("Processing with MAX Pooling...")
    for key, tensor in data.items():
        # Input shape: (1, 107, 4096)

        # STRATEGY: Max Pooling
        # We take the maximum value across the sequence dimension (dim=1)
        # This preserves strong signals (like errors) that might get averaged out
        # Result shape: (1, 4096)
        max_pooled = torch.max(tensor.float(), dim=1).values

        vectors.append(max_pooled.squeeze(0).numpy())

        # Extract Label
        label = key.split('_')[0] if '_' in key else 'Unknown'
        labels.append(label)

    X = np.array(vectors)
    print(f"Data Shape: {X.shape}")

    # 1. Scale Data (Crucial for UMAP/Cosine)
    print("Scaling...")
    X_scaled = StandardScaler().fit_transform(X)

    # 2. Run UMAP with Cosine Metric
    print("Running UMAP (Cosine)...")
    reducer = umap.UMAP(
        n_neighbors=15,   # Balances local vs global structure
        min_dist=0.1,     # How tightly to pack points
        n_components=2,
        metric='cosine',  # <--- The key change
        random_state=42
    )
    embedding = reducer.fit_transform(X_scaled)

    # 3. Plot
    print("Plotting...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=labels,
        palette="deep",
        s=60,
        alpha=0.8
    )

    plt.title("UMAP (Cosine Metric) with Max Pooling")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(args.output)
    print(f"Saved to {args.output}")