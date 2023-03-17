import numpy as np
import pandas as pd


def calculate_degree_measures(networks: dict) -> pd.DataFrame:
    all_measures = []

    for (seed, tick), network in networks.items():
        degrees = sorted([d for n, d in network.degree()], reverse=True)
        degree_mean = np.mean(degrees)
        degree_std = np.std(degrees, ddof=1)

        all_measures.append(
            {
                "SEED": seed,
                "TICK": tick,
                "DEGREES": degrees,
                "DEGREE_MEAN": degree_mean,
                "DEGREE_STD": degree_std,
            }
        )

    all_measures_df = pd.DataFrame(all_measures)

    return all_measures_df
