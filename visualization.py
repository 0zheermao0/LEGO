# coding=utf-8
import os
import json
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import argparse
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Argument parser for source and target datasets
parser = argparse.ArgumentParser(description='Visualization of LLM Selection, MoE Encoder, and t-SNE Embeddings')
parser.add_argument("--source", type=str, default='acmv9', help="Source dataset name")
parser.add_argument("--target", type=str, default='citationv1', help="Target dataset name")
parser.add_argument("--llm", type=str, default='qwen2.5:7b', help="LLM model name used for expert selection")
parser.add_argument("--output_dir", type=str, default='plots', help="Directory to save output plots")
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Define a green-based colormap for scientific visualization
green_cmap = sns.light_palette("seagreen", as_cmap=True)
# Custom green colormap for t-SNE scatter plot
greens = plt.get_cmap('Greens')
green_colormap = ListedColormap(greens(np.linspace(0.2, 0.9, 256)))

def load_llm_selections():
    """Load LLM expert selections from JSON file."""
    selections_path = f"log/{args.target}-{args.llm}-selections.json"
    if not os.path.exists(selections_path):
        raise FileNotFoundError(f"LLM selections file not found at {selections_path}")
    with open(selections_path, 'r') as f:
        selections = json.load(f)
    # Convert keys to integers
    selections = {int(k): v for k, v in selections.items()}
    return selections

def load_moe_expert_selections():
    """Load MoE expert selections from CSV file for the target dataset."""
    csv_path = f"log/{args.source}-{args.target}-expert-selection.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"MoE expert selection CSV not found at {csv_path}")
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            node_id = int(row[0])
            expert = int(row[1])
            prob = float(row[2])
            data.append((node_id, expert, prob))
    return data

def load_embeddings():
    """Load final embeddings and labels for target dataset."""
    embeddings_path = f"log/{args.source}-{args.target}-final-embeddings.pt"
    labels_path = f"log/{args.source}-{args.target}-final-labels.pt"
    if not os.path.exists(embeddings_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Embeddings or labels not found at {embeddings_path} or {labels_path}")
    embeddings = torch.load(embeddings_path).detach().numpy()
    labels = torch.load(labels_path).detach().numpy()
    return embeddings, labels

def plot_heatmap(data, title, filename, annot=True):
    """Plot heatmap with a green color scheme."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, cmap=green_cmap, annot=annot, fmt='.2f', cbar_kws={'label': 'Frequency/Probability'})
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Expert ID', fontsize=12)
    plt.ylabel('Node Group / Selection', fontsize=12)
    plt.tight_layout()
    output_path = os.path.join(args.output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {output_path}")

def plot_tsne(embeddings, labels, title, filename):
    """Plot t-SNE visualization with a green color scheme."""
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[green_colormap(label / len(unique_labels))], 
                   label=f'Class {label}', alpha=0.6, s=30)
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    output_path = os.path.join(args.output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot to {output_path}")

def create_llm_selection_heatmap(selections):
    """Create a heatmap for LLM expert selections."""
    # Count frequency of expert selections
    expert_counts = {}
    for node_id, expert in selections.items():
        expert_counts[expert] = expert_counts.get(expert, 0) + 1
    # Create a simple frequency matrix (1 row for all nodes)
    num_experts = max(expert_counts.keys(), default=0) + 1
    freq_matrix = np.zeros((1, num_experts))
    for expert, count in expert_counts.items():
        freq_matrix[0, expert] = count
    # Normalize to probabilities
    freq_matrix = freq_matrix / freq_matrix.sum()
    plot_heatmap(freq_matrix, 'LLM Expert Selection Distribution on Source Dataset', 'llm_selection_heatmap.png', annot=True)

def create_moe_encoder_heatmap(moe_data):
    """Create a heatmap for MoE encoder expert selections and probabilities."""
    # Group by node ID and expert to aggregate probabilities (if multiple k selections)
    df = pd.DataFrame(moe_data, columns=['NodeID', 'Expert', 'Probability'])
    # Pivot table for heatmap (average probability per expert per node group if needed)
    # For simplicity, assume k=1 or take the first selection per node
    pivot_prob = df.pivot_table(values='Probability', index='NodeID', columns='Expert', aggfunc='mean', fill_value=0)
    # Sample a subset if too many nodes for visualization
    if pivot_prob.shape[0] > 100:
        pivot_prob = pivot_prob.sample(n=100, random_state=42)
    plot_heatmap(pivot_prob, 'MoE Encoder Expert Assignment Probabilities on Target Dataset', 'moe_encoder_heatmap.png', annot=False)

def main():
    # Load data
    print("Loading LLM selections...")
    llm_selections = load_llm_selections()
    print("Loading MoE expert selections...")
    moe_data = load_moe_expert_selections()
    print("Loading embeddings and labels...")
    embeddings, labels = load_embeddings()
    
    # Plot heatmaps
    print("Generating LLM selection heatmap...")
    create_llm_selection_heatmap(llm_selections)
    print("Generating MoE encoder heatmap...")
    create_moe_encoder_heatmap(moe_data)
    
    # Plot t-SNE visualization
    print("Generating t-SNE visualization...")
    plot_tsne(embeddings, labels, 't-SNE Visualization of Encoder Embeddings on Target Dataset', 'tsne_embeddings.png')

if __name__ == "__main__":
    main()
