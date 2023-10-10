from torch_geometric.loader import NeighborLoader
def get_node_neighbors(data, node_type, node_ids, num_neighbors=[10000000]):
        input_nodes = (node_type, torch.LongTensor(node_ids))
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
                batch_size=1,
                shuffle=False,
                drop_last=True,
                num_workers=0,
                directed=True,  # contains only edges which are followed, False: contains full node induced subgraph
                #disjoint=True # sampled seed node creates its own, disjoint from the rest, subgraph, will add "batch vector" to loader output
                pin_memory=True, # faster data transfer to gpu
                #num_workers=2,
                #prefetch_factor=2
        )
        return next(iter(loader))