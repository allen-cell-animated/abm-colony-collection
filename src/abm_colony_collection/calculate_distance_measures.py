from typing import Union

import networkx as nx
import pandas as pd


def calculate_distance_measures(network: nx.Graph) -> pd.DataFrame:
    measures: list[dict[str, Union[int, float]]] = []

    for component in nx.connected_components(network):
        # Calculate eccentricity for connected subnetwork.
        eccentricity = nx.eccentricity(network.subgraph(component))

        # Extract distance measures for each node in subnetwork.
        measures = measures + [
            {
                "ID": node,
                "ECCENTRICITY": eccentricity[node],
            }
            for node in component
        ]

    return pd.DataFrame(measures)
