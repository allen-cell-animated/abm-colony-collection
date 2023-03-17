import networkx as nx
import numpy as np
import pandas as pd


def calculate_centrality_measures(networks: dict) -> pd.DataFrame:
    all_measures = []

    for (seed, tick), network in networks.items():
        degree_centralities = list(nx.degree_centrality(network).values())
        closeness_centralities = list(nx.closeness_centrality(network).values())
        betweenness_centralities = list(nx.betweenness_centrality(network).values())

        degree_centrality_mean = np.mean(degree_centralities)
        closeness_centrality_mean = np.mean(closeness_centralities)
        betweenness_centrality_mean = np.mean(betweenness_centralities)

        degree_centrality_std = np.std(degree_centralities, ddof=1)
        closeness_centrality_std = np.std(closeness_centralities, ddof=1)
        betweenness_centrality_std = np.std(betweenness_centralities, ddof=1)

        all_measures.append(
            {
                "SEED": seed,
                "TICK": tick,
                "DEGREE_CENTRALITY_MEAN": degree_centrality_mean,
                "CLOSENESS_CENTRALITY_MEAN": closeness_centrality_mean,
                "BETWEENNESS_CENTRALITY_MEAN": betweenness_centrality_mean,
                "DEGREE_CENTRALITY_STD": degree_centrality_std,
                "CLOSENESS_CENTRALITY_STD": closeness_centrality_std,
                "BETWEENNESS_CENTRALITY_STD": betweenness_centrality_std,
            }
        )

    all_measures_df = pd.DataFrame(all_measures)

    return all_measures_df
