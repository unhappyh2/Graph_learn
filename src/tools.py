import torch
from torch_geometric.utils import structured_negative_sampling
import random

def sample_mini_batch(batch_size, edge_index):
    """
    Args:
        batch_size (int): 批大小
        edge_index (torch.Tensor): 2*N的边列表
    Returns:
        tuple: user indices, positive item indices, negative item indices
    """
    edges = structured_negative_sampling(edge_index)
    edges = torch.stack(edges, dim=0)
    indices = random.choices(
        [i for i in range(edges[0].shape[0])], k=batch_size)
    batch = edges[:, indices]
    user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]
    return user_indices, pos_item_indices, neg_item_indices


def sample_items(items_emb, num_samples):
    indices = torch.randperm(items_emb.size(0))[:num_samples]
    return items_emb[indices]