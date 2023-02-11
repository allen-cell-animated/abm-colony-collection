import numpy as np
import pandas as pd
from prefect import task
from scipy.spatial import distance


@task
def calculate_cluster_distances(neighbors: pd.DataFrame) -> pd.DataFrame:
    all_clusters = []

    for (seed, tick), group in neighbors.groupby(["SEED", "TICK"]):
        clusters = group.groupby("GROUP").filter(lambda x: len(x.index) > 1)

        if clusters.empty:
            continue

        inter_distance_mean, inter_distance_std = calculate_inter_cluster_distances(clusters)
        intra_distance_mean, intra_distance_std = calculate_intra_cluster_distances(clusters)

        all_clusters.append(
            {
                "SEED": seed,
                "TICK": tick,
                "INTER_DISTANCE_MEAN": inter_distance_mean,
                "INTER_DISTANCE_STD": inter_distance_std,
                "INTRA_DISTANCE_MEAN": intra_distance_mean,
                "INTRA_DISTANCE_STD": intra_distance_std,
            }
        )

    all_clusters_df = pd.DataFrame(all_clusters)

    return all_clusters_df


def calculate_inter_cluster_distances(clusters: pd.DataFrame) -> tuple[float, float]:
    cluster_centroids = clusters.groupby("GROUP")[["cx", "cy", "cz"]].mean().values

    if cluster_centroids.shape[0] < 2:
        return (np.nan, np.nan)

    inter_distances = distance.cdist(cluster_centroids, cluster_centroids, "euclidean")
    distances = np.ndarray.flatten(inter_distances)
    distances = np.delete(distances, range(0, len(distances), len(inter_distances) + 1), 0)

    inter_distance_mean = np.mean(distances)
    inter_distance_std = np.std(distances, ddof=1)

    return (inter_distance_mean, inter_distance_std)


def calculate_intra_cluster_distances(clusters: pd.DataFrame) -> tuple[float, float]:
    intra_distance_means = []
    intra_distance_stds = []

    for _, group in clusters.groupby("GROUP"):
        cluster_centroids = group[["cx", "cy", "cz"]].values

        intra_distances = distance.cdist(cluster_centroids, cluster_centroids, "euclidean")
        distances = np.ndarray.flatten(intra_distances)
        distances = np.delete(distances, range(0, len(distances), len(intra_distances) + 1), 0)

        intra_distance_means.append(np.mean(distances))
        intra_distance_stds.append(np.std(distances, ddof=1))

    return (np.mean(intra_distance_means), np.mean(intra_distance_stds))
