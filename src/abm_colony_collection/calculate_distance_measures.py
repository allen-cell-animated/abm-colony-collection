import networkx as nx
import numpy as np
import pandas as pd
from prefect import task


@task
def calculate_distance_measures(networks: dict) -> pd.DataFrame:
    all_measures = []

    for (seed, tick), network in networks.items():
        if not nx.is_connected(network):
            subnetworks = [
                network.subgraph(component) for component in nx.connected_components(network)
            ]

            radius = np.mean([nx.radius(subnetwork) for subnetwork in subnetworks])
            diameter = np.mean([nx.diameter(subnetwork) for subnetwork in subnetworks])

            eccentricities = [nx.eccentricity(subnetwork) for subnetwork in subnetworks]
            eccentricity = np.mean([np.mean(list(ecc.values())) for ecc in eccentricities])

            shortest_paths = [
                nx.average_shortest_path_length(subnetwork) for subnetwork in subnetworks
            ]
            shortest_path = np.mean(shortest_paths)
        else:
            radius = nx.radius(network)
            diameter = nx.diameter(network)

            ecc = nx.eccentricity(network)
            eccentricity = np.mean(list(ecc.values()))

            shortest_path = nx.average_shortest_path_length(network)

        all_measures.append(
            {
                "SEED": seed,
                "TICK": tick,
                "RADIUS": radius,
                "DIAMETER": diameter,
                "ECCENTRICITY": eccentricity,
                "SHORTEST_PATH": shortest_path,
            }
        )

    all_measures_df = pd.DataFrame(all_measures)

    return all_measures_df
