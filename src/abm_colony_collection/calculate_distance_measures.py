from typing import Union

import networkx as nx
import pandas as pd


def calculate_distance_measures(network: nx.Graph) -> pd.DataFrame:
    measures: list[dict[str, Union[int, float]]] = []

    for component in nx.connected_components(network):
        eccentricity = nx.eccentricity(network.subgraph(component))

        measures = measures + [
            {
                "ID": node,
                "ECCENTRICITY": eccentricity[node],
            }
            for node in component
        ]

    measures_df = pd.DataFrame(measures)

    return measures_df
