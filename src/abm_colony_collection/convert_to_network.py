import networkx as nx
import pandas as pd


def convert_to_network(neighbors: pd.DataFrame) -> nx.Graph:
    """
    Convert lists of neighbors to a network object.

    Parameters
    ----------
    neighbors
        Lists of neighbors for each node id.

    Returns
    -------
    :
        The network object.
    """

    nodes = list(neighbors["ID"].values)
    edges = [
        (node_id, neighbor_id)
        for node_id, neighbor_ids in zip(neighbors["ID"], neighbors["NEIGHBORS"])
        for neighbor_id in neighbor_ids
    ]

    network = nx.Graph()
    network.add_nodes_from(nodes)
    network.add_edges_from(edges)

    return network
