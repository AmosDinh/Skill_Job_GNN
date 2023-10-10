from tqdm.auto import tqdm
import numpy as np
from torch_geometric.loader import NeighborLoader
import torch

def get_entity_embedding(model, data, node_type, num_neighbors, node_ids):
    input_nodes = (node_type, torch.LongTensor(node_ids))
    batch_size = 64
    loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            # {
            #     ('Job', 'REQUIRES', 'Skill'):num_neighbors,
            #     ('Skill', 'rev_REQUIRES', 'Job'):num_neighbors,
            #     ('Skill', 'IS_SIMILAR_SKILL', 'Skill'):num_neighbors, # In this example, index 0 will never be used, since neighboring edge to a job node can't be a skill-skill edge
            #     ('Skill', 'rev_IS_SIMILAR_SKILL', 'Skill'):num_neighbors,
            #     ('Job', 'IS_SIMILAR_JOB', 'Job'):num_neighbors,
            #     ('Job', 'rev_IS_SIMILAR_JOB', 'Job'):num_neighbors,
            # },
            input_nodes = input_nodes,
            #edge_label_index=(edge_type, data[edge_type].edge_label_index), # if (edge, None), None means all edges are considered
            #  =train_data[edge].edge_label,
            #neg_sampling=negative_sampling, # adds negative samples
            batch_size=min(len(node_ids),batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=0,
            directed=True,  # contains only edges which are followed, False: contains full node induced subgraph
            #disjoint=True # sampled seed node creates its own, disjoint from the rest, subgraph, will add "batch vector" to loader output
            pin_memory=True, # faster data transfer to gpu
            #num_workers=2,
            #prefetch_factor=2
    )
    num_norm_iterations = 20
    all_embeddings = []
    model.eval()
    for batch in tqdm(loader):
        embeddings = []
        for _ in range(num_norm_iterations):
            
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_weight_dict, batch.num_sampled_edges_dict, batch.num_sampled_nodes_dict)
            
            embeddings.append(out[node_type].reshape((1,out[node_type].shape[0],out[node_type].shape[1])).detach().cpu().numpy())
        
    
        embeddings = np.concatenate(embeddings, axis=0)
        # print(torch.std(embeddings,dim=0))
        embeddings = np.sum(embeddings, axis=0)/num_norm_iterations
        all_embeddings.append(embeddings)
    
    return np.concatenate(all_embeddings, axis=0)
