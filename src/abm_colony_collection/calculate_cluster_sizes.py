import pandas as pd


def calculate_cluster_sizes(neighbors: pd.DataFrame) -> pd.DataFrame:
    all_clusters = []

    for (seed, tick), group in neighbors.groupby(["SEED", "TICK"]):
        group_sizes = group.groupby("GROUP").size()
        cluster_sizes = group_sizes[group_sizes != 1]
        single_sizes = group_sizes[group_sizes == 1]

        all_clusters.append(
            {
                "SEED": seed,
                "TICK": tick,
                "NUM_CLUSTERS": len(cluster_sizes),
                "NUM_SINGLES": len(single_sizes),
                "CLUSTER_SIZE_TOTAL": cluster_sizes.sum(),
                "CLUSTER_SIZE_MEAN": cluster_sizes.mean(),
                "CLUSTER_SIZE_STD": cluster_sizes.std(ddof=1),
            }
        )

    all_clusters_df = pd.DataFrame(all_clusters)

    return all_clusters_df
