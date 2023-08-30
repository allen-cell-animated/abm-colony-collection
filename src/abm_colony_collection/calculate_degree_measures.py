import networkx as nx
import pandas as pd


def calculate_degree_measures(network: nx.Graph) -> pd.DataFrame:
    # Extract degree for each node in network.
    measures = [
        {
            "ID": node,
            "DEGREE": degree,
        }
        for node, degree in network.degree()
    ]

    return pd.DataFrame(measures)
