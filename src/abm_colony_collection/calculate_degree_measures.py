import networkx as nx
import pandas as pd


def calculate_degree_measures(network: nx.Graph) -> pd.DataFrame:
    """
    Calculate degree measures for each node in network.

    Measures include:

    - Degree = number of edges adjacent to the node

    Parameters
    ----------
    network
        The network object.

    Returns
    -------
    :
        Degree measures for each node in the network.
    """

    # Extract degree for each node in network.
    measures = [
        {
            "ID": node,
            "DEGREE": degree,
        }
        for node, degree in network.degree()
    ]

    return pd.DataFrame(measures)
