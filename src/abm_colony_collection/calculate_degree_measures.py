import networkx as nx
import pandas as pd


def calculate_degree_measures(network: nx.Graph) -> pd.DataFrame:
    measures = [
        {
            "ID": node,
            "DEGREE": degree,
        }
        for node, degree in network.degree()
    ]

    measures_df = pd.DataFrame(measures)

    return measures_df
