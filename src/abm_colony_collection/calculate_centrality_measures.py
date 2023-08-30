import networkx as nx
import pandas as pd


def calculate_centrality_measures(network: nx.Graph) -> pd.DataFrame:
    # Calculate different centrality measures for network.
    degree_centralities = nx.degree_centrality(network)
    closeness_centralities = nx.closeness_centrality(network)
    betweenness_centralities = nx.betweenness_centrality(network)

    # Extract centrality measures for each node in network.
    measures = [
        {
            "ID": node,
            "DEGREE_CENTRALITY": degree_centralities[node],
            "CLOSENESS_CENTRALITY": closeness_centralities[node],
            "BETWEENNESS_CENTRALITY": betweenness_centralities[node],
        }
        for node in network.nodes
    ]

    return pd.DataFrame(measures)
