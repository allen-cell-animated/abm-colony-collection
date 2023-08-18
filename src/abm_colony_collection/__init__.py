"""Tasks for analyzing colony dynamics including clustering and graph measures."""

from prefect import task

from .calculate_centrality_measures import calculate_centrality_measures
from .calculate_cluster_distances import calculate_cluster_distances
from .calculate_cluster_sizes import calculate_cluster_sizes
from .calculate_degree_measures import calculate_degree_measures
from .calculate_distance_measures import calculate_distance_measures
from .convert_to_network import convert_to_network
from .get_depth_map import get_depth_map
from .get_neighbors_map import get_neighbors_map
from .make_voxels_array import make_voxels_array

calculate_centrality_measures = task(calculate_centrality_measures)
calculate_cluster_distances = task(calculate_cluster_distances)
calculate_cluster_sizes = task(calculate_cluster_sizes)
calculate_degree_measures = task(calculate_degree_measures)
calculate_distance_measures = task(calculate_distance_measures)
convert_to_network = task(convert_to_network)
get_depth_map = task(get_depth_map)
get_neighbors_map = task(get_neighbors_map)
make_voxels_array = task(make_voxels_array)
