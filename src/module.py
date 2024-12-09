import torch
from torch import nn
from torch_geometric.nn import MessagePassing

class GNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__(aggr='sum', flow='source_to_target')
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.sigmoid = torch.nn.Sigmoid()
        self.matrix = nn.Parameter(torch.randn(in_channels, in_channels, requires_grad=True))

    def forward(self, x_src,x_tgt,edge_index):
        
        out = self.propagate(x_src=x_src, x_tgt=x_tgt,edge_index=edge_index)
        #out = self.linear(out)
        return out
    
    def message(self, x_src,edge_index):
        message_src = x_src[edge_index[0]]
        return message_src
    
    def aggregate(self, inputs, edge_index, dim_size=None):
        weights = self.calculate_weights(edge_index[1])
        weights = weights.view(-1, 1).expand_as(inputs)
        inputs = inputs * weights

        return super().aggregate(inputs, edge_index[1])
    

    def update(self, aggr_out):
        aggr_out = torch.matmul(aggr_out, self.matrix.t())
        aggr_out = self.sigmoid(aggr_out)
        return aggr_out
    
    def calculate_weights(self,index):
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
    
class GModel(MessagePassing):
    def __init__(self, in_channels, out_channels,k=3):
        super(GModel, self).__init__(aggr='sum', flow='source_to_target')
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.sigmoid = torch.nn.Sigmoid()
        self.matrix = nn.Parameter(torch.randn(in_channels, in_channels, requires_grad=True))
        self.k = k

    def forward(self, x_src,x_tgt,edge_index,edge_attr):
        edge_attr = self.sigmoid(edge_attr-1)
        out = self.propagate(x_src=x_src, x_tgt=x_tgt,edge_index=edge_index,edge_attr=edge_attr)
        return out

    def message(self, x_src,edge_index,edge_attr):
        message_src = x_src[edge_index[0]]
        message_src = message_src * edge_attr.view(-1,1)
        return message_src

    def aggregate(self, inputs, edge_index, dim_size=None):
        weights = self.calculate_weights(edge_index[1])
        weights = weights.view(-1, 1).expand_as(inputs)
        inputs = inputs * weights

        return super().aggregate(inputs, edge_index[1])


    def update(self, aggr_out):
        aggr_out = torch.matmul(aggr_out, self.matrix.t())
        aggr_out = self.sigmoid(aggr_out)
        return aggr_out

    def calculate_weights(self,index):
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