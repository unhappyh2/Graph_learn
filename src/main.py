from datapross import DataProcess, SequenceEncoder, GenresEncoder, IdentityEncoder
from tools import *
from data_evaluation.loss_function import bpr_loss
from data_evaluation.evaluation import evaluation
from module import GNN,GModel
import pandas as pd
from torch_geometric.data import HeteroData
import torch
from tqdm import trange
from torch import nn
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

movie_path = '../data/ml-latest-small/movies.csv'
rating_path = '../data/ml-latest-small/ratings.csv'
print(pd.read_csv(movie_path).head())
print(pd.read_csv(rating_path).head())

data_process = DataProcess()
_, users_mapping = data_process.load_node_csv(rating_path, index_col='userId')
items_x, items_mapping = data_process.load_node_csv(
    movie_path, index_col='movieId', encoders={
        'title': SequenceEncoder(),
        'genres': GenresEncoder()
    })

edge_index, edge_label = data_process.load_edge_csv(
    rating_path,
    src_index_col='movieId',
    src_mapping=items_mapping,
    dst_index_col='userId',
    dst_mapping=users_mapping,
    encoders={'rating': IdentityEncoder(dtype=torch.long)},
)

data = HeteroData()
data['users'].mapping = users_mapping
data['items'].x = items_x
data['items'].mapping = items_mapping
data['edge'].edge_index = edge_index
data['edge'].edge_label = edge_label

num_node = len(users_mapping)+len(items_mapping)
user_emb = nn.Embedding(num_embeddings=len(users_mapping)+1, embedding_dim=items_x.size(dim=1))
nn.init.normal_(user_emb.weight, std=0.1)
user_x=torch.arange(1,len(users_mapping)+1).view(1,-1)
user_x = torch.LongTensor(user_x)
user_x = user_emb(user_x)
user_x = user_x.squeeze(0)
data['users'].x = user_x

embedding_dim = items_x.size(dim=1)
model = GModel(in_channels=embedding_dim, out_channels=embedding_dim,k=3)
model = model.to(device)
data = data.to(device)

learning_rate = 1e-2
batch_size = 64
iteration = 5000
lambda_val = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
losses = []

for iter in trange(iteration):
    x_src, x_tgt, edge_index,edge_attr = data['items'].x, data['users'].x,data['edge'].edge_index,data['edge'].edge_label
    users_emb_final = model.forward(x_src,x_tgt,edge_index,edge_attr)
    edge_index_t = torch.flip(data['edge'].edge_index, dims=[0])
    edges = structured_negative_sampling(edge_index_t)
    edges = torch.stack(edges, dim=0)
    num_users = users_emb_final.size(0)
    users_emb_final = users_emb_final[edges[0]]  
    users_emb_0 = data['users'].x[edges[0]]
    pos_items_emb_0 = data['items'].x[edges[1]]
    neg_items_emb_0 = data['items'].x[edges[2]]
    pos_items_emb_final = pos_items_emb_0
    neg_items_emb_final = neg_items_emb_0
    bpr_loss_value = bpr_loss(users_emb_final,users_emb_0,pos_items_emb_final,pos_items_emb_final,
                              neg_items_emb_final,neg_items_emb_final,lambda_val)
    # 记录损失
    losses.append(bpr_loss_value.item())
    bpr_loss_value.backward(retain_graph=True)
    optimizer.step()
    optimizer.zero_grad()
    if iter % 100 == 0:
        print("loss = ",bpr_loss_value.item())
print("Train done !")

plt.plot(losses)
plt.show()





