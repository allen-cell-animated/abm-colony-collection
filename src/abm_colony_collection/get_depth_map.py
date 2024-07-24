import numpy as np
from scipy import ndimage
from skimage import measure


def get_depth_map(array: np.ndarray, neighbors_map: dict) -> dict:
    """
    Get map of id to depth starting from depth = 1 at the edge if the region.

    All ids at the edge of the array are assigned a depth of 1. Immediate
    neighbors of those edges are assigned a depth of 2, and so on, until all ids
    have an assigned depth.

    Parameters
    ----------
    array
        Segmentation array.
    neighbors_map
        Map of ids to lists of neighbors.

    Returns
    -------
    :
        Map of id to depth from edge.
    """

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
    """
    Gets ids of regions closest to the edge of the array.

    Parameters
    ----------
    array
        Segmentation array.

    Returns
    -------
    :
        List of edge arrays.
    """

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
