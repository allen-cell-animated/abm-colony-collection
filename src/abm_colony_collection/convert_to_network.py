import networkx as nx
import pandas as pd
from prefect import task


@task
def convert_to_network(neighbors: pd.DataFrame) -> nx.Graph:
    networks = {}

    for tick, group in neighbors.groupby("TICK"):
        nodes = list(group["ID"].values)
        edges = [
            (node_id, neighbor_id)
            for node_id, neighbor_ids in zip(group["ID"], group["NEIGHBORS"])
            for neighbor_id in neighbor_ids
        ]

        network = nx.Graph()
        network.add_nodes_from(nodes)
        network.add_edges_from(edges)

        networks[tick] = network

    return networks
