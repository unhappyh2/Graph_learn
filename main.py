#%%
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
#%%
from torch_geometric.data import download_url, extract_zip

#url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
#extract_zip(download_url(url, '.'), './data')
movie_path = './data/ml-latest-small/movies.csv'
rating_path = './data/ml-latest-small/ratings.csv'
#%%
import pandas as pd

print(pd.read_csv(movie_path).head())
print(pd.read_csv(rating_path).head())
#%%
import torch

def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping
#%%
from sentence_transformers import SentenceTransformer
class SequenceEncoder:
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()
#%%
class GenresEncoder:
    def __init__(self, sep='|'):
        self.sep = sep

    def __call__(self, df):
        genres = set(g for col in df.values for g in col.split(self.sep))
        mapping = {genre: i for i, genre in enumerate(genres)}

        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for genre in col.split(self.sep):
                x[i, mapping[genre]] = 1
        return x
#%%
movie_x, movie_mapping = load_node_csv(
    movie_path, index_col='movieId', encoders={
        'title': SequenceEncoder(),
        'genres': GenresEncoder()
    })
#%%
user_x, user_mapping = load_node_csv(rating_path, index_col='userId')
#%%
from torch_geometric.data import HeteroData

data = HeteroData()

data['user'].num_nodes = len(user_mapping)  # Users do not have any features.
data['movie'].x = movie_x

print(data)
#%%
def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr
#%%
class IdentityEncoder:
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)
#%%
edge_index, edge_label = load_edge_csv(
    rating_path,
    src_index_col='userId',
    src_mapping=user_mapping,
    dst_index_col='movieId',
    dst_mapping=movie_mapping,
    encoders={'rating': IdentityEncoder(dtype=torch.long)},
)

data['user', 'rates', 'movie'].edge_index = edge_index
data['user', 'rates', 'movie'].edge_label = edge_label

#%%
from torch import nn ,optim, Tensor
num_node = len(user_mapping)+len(movie_mapping)
user_emb = nn.Embedding(num_embeddings=len(user_mapping)+1, embedding_dim=movie_x.size(dim=1))
nn.init.normal_(user_emb.weight, std=0.1)
user_x=torch.arange(1,len(user_mapping)+1).view(1,-1)
user_x = torch.LongTensor(user_x)
user_x = user_emb(user_x)
user_x = user_x.squeeze(0)
#%%
from torch_geometric.utils import structured_negative_sampling
#edge_index_new = structured_negative_sampling(edge_index)
#edge_index_new = torch.stack(edge_index_new, dim=0)
#%%
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
edge_index_samp = sample_mini_batch(64,edge_index)


#%%
import torch
from torch_geometric.seed import seed_everything
from torch_geometric.nn import MessagePassing
seed_everything(11)
from torch_geometric.utils import add_self_loops, degree
#%%
# 定义GCN空域图卷积神经网络
class GCNConv(MessagePassing):
    # 网络初始化
    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: 节点属性向量的维度
        :param out_channels: 经过图卷积之后，节点的特征表示维度
        """
        # 定义伽马函数为求和函数,aggr='add'
        super(GCNConv, self).__init__(aggr='add')
        # 定义最里面那个线性变换
        # 具体到实现中就是一个线性层
        self.linear_change = torch.nn.Linear(in_channels, out_channels)
 
    # 定义信息汇聚函数
    def message(self, x_j, norm):
        # 正则化
        # norm.view(-1,1)将norm变为一个列向量
        # x_j是节点的特征表示矩阵
        return norm.view(-1, 1) * x_j
 
    # 前向传递，进行图卷积
    def forward(self, x, edge_index):
        """
        :param x:图中的节点，维度为[节点数,节点属性相邻维度数]
        :param edge_index: 图中边的连接信息,维度为[2,边数]
        :return:
        """
        # 添加节点到自身的环
        # 因为节点最后面汇聚相邻节点信息时包含自身
        # add_self_loops会在edge_index边的连接信息表中，
        # 添加形如[i,i]这样的信息
        # 表示一个节点到自身的环
        # 函数返回[边的连接信息，边上的属性信息]
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # 进行线性变换
        x = self.linear_change(x)
        # 计算外面的正则化
        row, col = edge_index
        # 获取节点的度
        deg = degree(col, x.size(0), dtype=x.dtype)
        # 带入外面的正则化公式
        deg_inv_sqrt = deg.pow(-0.5)
        # 将未知的值设为0，避免下面计算出错
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # 正则化部分
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # 进行信息传递和融合
        # propagate会自动调用self.message函数，并将参数传递给它
        return self.propagate(edge_index, x=x, norm=norm)

#%%
# 实例化一个图卷积神经网络
# 并假设图节点属性向量的维度为16，图卷积出来的节点特征表示向量维度为32
conv = GCNConv(404, 32)
# 随机生成一个节点属性向量
# 5个节点，属性向量为16维
# 随机生成边的连接信息
# 假设有3条边
edge_index = torch.tensor(edge_index, dtype=torch.long)
# 进行图卷积
output = conv(movie_x, edge_index)
# 输出卷积之后的特征表示矩阵

#%%
learning_rate = 1e-3
batch_size = 64
epochs = 5
ITERATIONS = 256
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(GCNConv.parameters(), learning_rate)
for iter in ITERATIONS:
    
    output = conv(movie_x, edge_index)
    loss = loss_fn(pred, y)