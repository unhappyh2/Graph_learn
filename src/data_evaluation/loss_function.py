import torch

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