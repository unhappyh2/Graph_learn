import torch

def calculate_weights(index):
    """
    计算每个索引位置的权重（根据相同元素的数量）
    Args:
        index: 索引数组，如 [0,0,0,1,1,2,2,3]
    Returns:
        weights: 权重数组，如 [0.33,0.33,0.33,0.5,0.5,0.5,0.5,1]
    """
    # 使用 unique_counts 获取每个元素的计数
    unique_elements, counts = torch.unique(index, return_counts=True)
    
    # 创建一个和输入相同大小的权重张量
    weights = torch.zeros_like(index, dtype=torch.float)
    
    # 为每个元素分配权重
    for element, count in zip(unique_elements, counts):
        weights[index == element] = 1.0 / count.float()
    
    return weights

index = torch.tensor([0,0,0,1,1,2,2,3])
weights = calculate_weights(index)
print(weights)

