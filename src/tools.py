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

def bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, lambda_val):
    """
    Args:
        users_emb_final (torch.Tensor): e_u^k
        users_emb_0 (torch.Tensor): e_u^0
        pos_items_emb_final (torch.Tensor): positive e_i^k
        pos_items_emb_0 (torch.Tensor): positive e_i^0
        neg_items_emb_final (torch.Tensor): negative e_i^k
        neg_items_emb_0 (torch.Tensor): negative e_i^0
        lambda_val (float): λ的值
    Returns:
        torch.Tensor: loss值
    """
    reg_loss = lambda_val * (users_emb_0.norm(2).pow(2) +
                             pos_items_emb_0.norm(2).pow(2) +
                             neg_items_emb_0.norm(2).pow(2))  # L2 loss L2范数是指向量各元素的平方和然后求平方根

    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1) # 正采样预测分数
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1) # 负采样预测分数

    loss = -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores)) + reg_loss

    return loss

def sample_items(items_emb, num_samples):
    indices = torch.randperm(items_emb.size(0))[:num_samples]
    return items_emb[indices]