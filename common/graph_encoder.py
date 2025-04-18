# coding=utf-8
import torch
from collections import defaultdict

class GraphEncoder:
    def __init__(self):
        pass
    
    def encode(self, edge_index, mask=None, num_nodes=None):
        """
        Encode graph data into a textual description.
        
        Args:
            edge_index (torch.Tensor): Tensor of shape (2, E) representing edges in the graph.
            mask (torch.Tensor, optional): Boolean tensor of shape (N,) indicating which nodes to include.
            num_nodes (int, optional): Number of nodes in the graph. If not provided, inferred from edge_index.
        
        Returns:
            str: Textual description of the graph structure for unmasked nodes.
        """
        if num_nodes is None:
            num_nodes = edge_index.max().item() + 1 if edge_index.numel() > 0 else 0
            
        # Build adjacency list
        adj_dict = defaultdict(list)
        for src, dst in edge_index.t().tolist():
            adj_dict[src].append(dst)
            adj_dict[dst].append(src)  # Since edges are bidirectional
        
        # Deduplicate and sort neighbors
        for node in adj_dict:
            adj_dict[node] = sorted(list(set(adj_dict[node])))
        
        # Determine nodes to describe
        if mask is not None:
            nodes_to_describe = [i for i in range(num_nodes) if mask[i].item()]
        else:
            nodes_to_describe = list(range(num_nodes))
        
        if not nodes_to_describe:
            return "G describes an empty graph with no visible nodes."
            
        # Build description
        description = f"G describes a graph among {', '.join(map(str, nodes_to_describe))}. In this graph:"
        for node in nodes_to_describe:
            neighbors = adj_dict.get(node, [])
            if neighbors:
                description += f"\n- Node {node} is connected to nodes {', '.join(map(str, neighbors))}."
            else:
                description += f"\n- Node {node} has no connections."
                
        return description
