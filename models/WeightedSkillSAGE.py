from torch_geometric.nn.conv import SimpleConv
from typing import List, Optional, Union, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import add_self_loops, spmm

class WeightedGraphSageConv(SimpleConv):
    def __init__(self, in_channels: Union[int, Tuple[int, int]], out_channels: int, normalize, max_pool, combine_root: Optional[str] = None, batch_norm=False, activation='relu', aggr: str = 'add', bias: bool = True, **kwargs):
        super().__init__(aggr, combine_root, **kwargs)
        
        # from GraphConv https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/graph_conv.html#GraphConv
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
            
        if self.combine_root == 'cat':
            assert out_channels%2==0 and out_channels!=-1, 'number of in_channels must be even, and no lazy initialization (-1) is supported'
            to_out_channels = out_channels//2
        else:
            to_out_channels = out_channels
            
        
        
        self.normalize = normalize
        
        if max_pool:
            self.max_pool = True
            self.lin_for_max_pool = Linear(in_channels[0], out_channels, bias=True)
            in_ch = out_channels
        else:
            in_ch = in_channels[0]
            
        
        
        if batch_norm:
            self.batch_norm = True
            self.batchnorm = BatchNorm(in_channels=in_ch)
        
        if activation == 'prelu':
            self.activation = torch.nn.PReLU(num_parameters=out_channels)
            
            self.max_pool_activation = torch.nn.PReLU(num_parameters=in_ch)
      
                
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
            self.max_pool_activation = torch.nn.ReLU()
        
        
        self.lin_i_out = Linear(in_channels[0], to_out_channels, bias=bias)
        self.lin_j_out = Linear(in_ch, to_out_channels, bias=bias)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        super().reset_parameters()
        self.lin_j_out.reset_parameters()
        self.lin_i_out.reset_parameters()
        self.lin_for_max_pool.reset_parameters()
        self.batchnorm.reset_parameters()
        # self.activation.reset_parameters()
    
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:

        if self.combine_root is not None:
            if self.combine_root == 'self_loop':
                if not isinstance(x, Tensor) or (size is not None
                                                 and size[0] != size[1]):
                    raise ValueError("Cannot use `combine_root='self_loop'` "
                                     "for bipartite message passing")
                if isinstance(edge_index, Tensor):
                    edge_index, edge_weight = add_self_loops(
                        edge_index, edge_weight, num_nodes=x.size(0))
                elif isinstance(edge_index, SparseTensor):
                    edge_index = torch_sparse.set_diag(edge_index)

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.max_pool:
            # just linear layer
            x0 = self.lin_for_max_pool(x[0])
            if self.batch_norm:
                x0 = self.batchnorm(x0)
            x = (self.max_pool_activation(x0), x[1])
            
        elif self.batch_norm:
            x = (self.batchnorm(x[0]), x[1])
        
        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)

        x_dst = x[1]
        if x_dst is not None and self.combine_root is not None and self.combine_root!='self_loop':
            x_dst = self.lin_i_out(x_dst)
            out = self.lin_j_out(out)
            if self.combine_root == 'sum':
                out = out + x_dst
            elif self.combine_root == 'cat':
                out = torch.cat([x_dst, out], dim=-1)
        
        out = self.activation(out)
        
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)
        return out


# Sage conv from the paper, max pooling would be
# normalize will set length
# you can choose to pass edgeweights, then it wont be exactly as in the paper
#conv = WeightedGraphSageConv(256, 256, normalize=True, combine_root='cat', aggr='max', bias=True, max_pool=True)

from typing import Tuple, Union
from torch import Tensor
import torch
import torch_geometric
from torch_geometric.nn import to_hetero, HeteroDictLinear, Linear
from torch_geometric.nn.conv import GraphConv, SAGEConv, SimpleConv, HeteroConv

from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric import seed_everything
from torch_geometric.utils import trim_to_layer

class WeightedSkillSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_conv_layers):
        super().__init__()
        
        self.lin_in = torch.nn.ModuleDict({
            'Skill': Linear(in_channels,hidden_channels),
            'Job': Linear(in_channels,hidden_channels)
        })
        
        
      
        
        self.hetero_convs = torch.nn.ModuleList()
        for i in range(n_conv_layers):
            # if i == n_conv_layers-1:
            #     in_ch = (in_channels, in_channels)
            # else:
         
            in_ch = (hidden_channels, hidden_channels)
                
            skill_skill = WeightedGraphSageConv(in_ch, hidden_channels, normalize=True, max_pool=True, combine_root='cat', aggr='max', bias=True)  # use same for rev_skill as well
            job_job = WeightedGraphSageConv(in_ch, hidden_channels, normalize=True, max_pool=True, combine_root='cat', aggr='max', bias=True)  # use same for rev_job... as well
            conv = HeteroConv(
                {
                    ('Job', 'REQUIRES', 'Skill'): WeightedGraphSageConv(in_ch, hidden_channels, normalize=True, max_pool=True, combine_root='cat', aggr='max', bias=True),
                    ('Skill', 'rev_REQUIRES', 'Job'): WeightedGraphSageConv(in_ch, hidden_channels, normalize=True, max_pool=True, combine_root='cat', aggr='max', bias=True),
                    ('Skill', 'IS_SIMILAR_SKILL', 'Skill'):skill_skill,
                    ('Skill', 'rev_IS_SIMILAR_SKILL', 'Skill'):skill_skill,
                    ('Job', 'IS_SIMILAR_JOB', 'Job'):job_job,
                    ('Job', 'rev_IS_SIMILAR_JOB', 'Job'):job_job,
                }, aggr='sum')
            self.hetero_convs.append(conv)
            
        self.lin_out = torch.nn.ModuleDict({
            'Skill': Linear(hidden_channels, out_channels),
            'Job': Linear(hidden_channels, out_channels)
        })

    def forward(self, x_dict, edge_index_dict, edge_weight_dict, num_sampled_edges_dict, num_sampled_nodes_dict):
        x_dict = {key: F.relu(self.lin_in[key](x)) for key, x in x_dict.items()}
        
        # speedup: only compute necessary node representations in each pass through https://pytorch-geometric.readthedocs.io/en/latest/advanced/hgam.html
        for i, conv in enumerate(self.hetero_convs):
            x_dict, edge_index_dict, edge_weight_dict = trim_to_layer(
                layer=i,
                num_sampled_nodes_per_hop=num_sampled_nodes_dict, 
                num_sampled_edges_per_hop=num_sampled_edges_dict, # gives the num sampled edges per edge type, e.g. ('Job', 'REQUIRES', 'Skill'): [3083, 14514] -> 3000 in first step, 14000 in second
                x=x_dict,
                edge_index=edge_index_dict,
                edge_attr=edge_weight_dict
            )
        
            x_dict = conv(x_dict, edge_index_dict, edge_weight_dict) # edge_weight_dict
            # x_dict = {key: F.relu(x) for key, x in x_dict.items()} # relu already implemented
            
        
        x_dict = {key: F.relu(self.lin_out[key](x)) for key, x in x_dict.items()}
        return x_dict



def weightedSkillSAGE_lr_2emin7_1lin_1lin_256dim_edgeweight_checkpoints():
    seed_everything(14)
    # this one has num_neighbors =[5,4] in the link neighbor loader
    model = WeightedSkillSAGE(in_channels=132, hidden_channels=256, out_channels=256, n_conv_layers=2)
    return model


def weightedSkillSAGE_lr_2emin7_1lin_1lin_256dim_edgeweight_noskillskillpred_checkpoints():
    seed_everything(14)
    # this one has num_neighbors =[5,4] in the link neighbor loader
    # the model was not trained on predicting skillskill edges, only s-j and j-j
    model = WeightedSkillSAGE(in_channels=132, hidden_channels=256, out_channels=256, n_conv_layers=2)
    return model


class WeightedSkillSAGEPreluBatchnorm(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_conv_layers):
        super().__init__()
        
        # self.lin_in1 = torch.nn.ModuleDict({
        #     'Skill': Linear(in_channels,hidden_channels),
        #     'Job': Linear(in_channels,hidden_channels)
        # })
        # self.lin_in2 = torch.nn.ModuleDict({
        #     'Skill': Linear(in_channels,hidden_channels),
        #     'Job': Linear(in_channels,hidden_channels)
        # })
        
        
        self.hetero_convs = torch.nn.ModuleList()
        for i in range(n_conv_layers):
            if i == 0:
                in_ch = (in_channels, in_channels)
            else:
                in_ch = (hidden_channels, hidden_channels)
            
            if i == n_conv_layers-1:
                out_ch = out_channels
            else:
                out_ch = hidden_channels
            
            skill_skill = WeightedGraphSageConv(in_ch, out_ch, normalize=True, max_pool=True, batch_norm=True, activation='prelu', combine_root='cat', aggr='max', bias=True)  # use same for rev_skill as well
            job_job = WeightedGraphSageConv(in_ch, out_ch, normalize=True, max_pool=True, batch_norm=True, activation='prelu', combine_root='cat', aggr='max', bias=True)  # use same for rev_job... as well
            conv = HeteroConv(
                {
                    ('Job', 'REQUIRES', 'Skill'): WeightedGraphSageConv(in_ch, out_ch, normalize=True, max_pool=True, batch_norm=True, activation='prelu', combine_root='cat', aggr='max', bias=True),
                    ('Skill', 'rev_REQUIRES', 'Job'): WeightedGraphSageConv(in_ch, out_ch, normalize=True, max_pool=True, batch_norm=True, activation='prelu', combine_root='cat', aggr='max', bias=True),
                    ('Skill', 'IS_SIMILAR_SKILL', 'Skill'):skill_skill,
                    ('Skill', 'rev_IS_SIMILAR_SKILL', 'Skill'):skill_skill,
                    ('Job', 'IS_SIMILAR_JOB', 'Job'):job_job,
                    ('Job', 'rev_IS_SIMILAR_JOB', 'Job'):job_job,
                }, aggr='sum')
            self.hetero_convs.append(conv)
            
        # self.lin_out1 = torch.nn.ModuleDict({
        #     'Skill': Linear(hidden_channels, out_channels),
        #     'Job': Linear(hidden_channels, out_channels)
        # })
        # self.lin_out2 = torch.nn.ModuleDict({
        #     'Skill': Linear(hidden_channels, out_channels),
        #     'Job': Linear(hidden_channels, out_channels)
        # })

    def forward(self, x_dict, edge_index_dict, edge_weight_dict, num_sampled_edges_dict, num_sampled_nodes_dict):
        # x_dict = {key: F.relu(self.lin_in[key](x)) for key, x in x_dict.items()}
        
        # speedup: only compute necessary node representations in each pass through https://pytorch-geometric.readthedocs.io/en/latest/advanced/hgam.html
        for i, conv in enumerate(self.hetero_convs):
            x_dict, edge_index_dict, edge_weight_dict = trim_to_layer(
                layer=i,
                num_sampled_nodes_per_hop=num_sampled_nodes_dict, 
                num_sampled_edges_per_hop=num_sampled_edges_dict, # gives the num sampled edges per edge type, e.g. ('Job', 'REQUIRES', 'Skill'): [3083, 14514] -> 3000 in first step, 14000 in second
                x=x_dict,
                edge_index=edge_index_dict,
                edge_attr=edge_weight_dict
            )
        
            x_dict = conv(x_dict, edge_index_dict, edge_weight_dict) # edge_weight_dict
            # x_dict = {key: F.relu(x) for key, x in x_dict.items()} # relu already implemented
            
        
        # x_dict = {key: F.relu(self.lin_out[key](x)) for key, x in x_dict.items()}
        return x_dict


def weightedSkillSAGE_lr_2emin7_0lin_256dim_edgeweight_prelu_batchnorm_checkpoints():
    seed_everything(14)
    # this one has num_neighbors =[5,4] in the link neighbor loader
    # the model was not trained on predicting skillskill edges, only s-j and j-j
    model = WeightedSkillSAGEPreluBatchnorm(in_channels=132, hidden_channels=256, out_channels=256, n_conv_layers=2)
    return model


def weightedSkillSAGE_lr_2emin7_0lin_132dim_edgeweight_prelu_batchnorm_checkpoints():
    seed_everything(14)
    # this one has num_neighbors =[5,4] in the link neighbor loader
    # the model was not trained on predicting skillskill edges, only s-j and j-j
    model = WeightedSkillSAGEPreluBatchnorm(in_channels=132, hidden_channels=132, out_channels=132, n_conv_layers=2)
    return model

def skillsage_388_prelu_batchnorm_edgeweight():
    seed_everything(14)
    # this one has num_neighbors =[5,4] in the link neighbor loader
    # the model was not trained on predicting skillskill edges, only s-j and j-j
    model = WeightedSkillSAGEPreluBatchnorm(in_channels=388, hidden_channels=256, out_channels=256, n_conv_layers=2)
    return model
    