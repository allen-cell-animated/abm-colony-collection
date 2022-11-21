from prefect import task
import numpy as np
from skimage import measure
from scipy import ndimage


@task
def get_depth_map(array: np.ndarray, neighbors_map: dict) -> dict:
    depth_map = {cell_id: 0 for cell_id in np.unique(array)}
    depth_map.pop(0, None)

    edge_ids = find_edge_ids(array)
    visited = set(edge_ids)
    queue = edge_ids.copy()

    while queue:
        current_id = queue.pop(0)

        current_neighbors = neighbors_map[current_id]["neighbors"]
        valid_neighbors = set(current_neighbors) - visited
        visited.update(valid_neighbors)
        queue = queue + list(valid_neighbors)

        for neighbor_id in valid_neighbors:
            depth_map[neighbor_id] = depth_map[current_id] + 1

        depth_map[current_id] = depth_map[current_id] + 1

    return depth_map


def find_edge_ids(array: np.ndarray) -> list[int]:
    slice_index = np.argmax(np.count_nonzero(array, axis=(1, 2)))
    array_slice = array[slice_index, :, :]

    # Calculate voronoi from cell shapes.
    distances = ndimage.distance_transform_edt(
        array_slice == 0, return_distances=False, return_indices=True
    )
    distances = distances.astype("uint16", copy=False)
    coordinates_y = distances[0].flatten()
    coordinates_x = distances[1].flatten()
    voronoi = array_slice[coordinates_y, coordinates_x].reshape(array_slice.shape)

    # Create border mask.
    mask = np.zeros(array_slice.shape, dtype="uint8")
    mask[array_slice != 0] = 1
    while measure.euler_number(mask) != 1:
        mask = ndimage.binary_dilation(mask, iterations=1)

    # Filter voronoi by mask to get edge ids.
    voronoi[mask == 1] = 0
    edge_ids = list(np.unique(voronoi))
    edge_ids.remove(0)

    return edge_ids
