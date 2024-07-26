from typing import Union

import networkx as nx
import pandas as pd


def calculate_distance_measures(network: nx.Graph) -> pd.DataFrame:
    """
    Calculate distance measures for each node in network.

    Measures include:

    - Eccentricity = maximum distance from node to all other nodes

    Parameters
    ----------
    network
        The network object.

    Returns
    -------
    :
        Distance measures for each node in the network.
    """

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
