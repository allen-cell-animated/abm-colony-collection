import numpy as np


def make_voxels_array(locations: list) -> np.ndarray:
    """
    Convert list of locations to a segmentated image array.

    Parameters
    ----------
    locations
        List of location dictionaries containing id and voxels.

    Returns
    -------
    :
        Segmentation array.
    """

    # Extract all voxel positions with id.
    all_ids: list[int] = []
    all_xyz: list[tuple[int, int, int]] = []
    for location in locations:
        cell_id = location["id"]
        xyz = [(x, y, z) for region in location["location"] for x, y, z in region["voxels"]]
        all_xyz = all_xyz + xyz
        all_ids = all_ids + [cell_id] * len(xyz)

    # Return if no voxels.
    if len(all_ids) == 0:
        return np.zeros((1, 1, 1))

    # Create empty array.
    mins = np.min(all_xyz, axis=0)
    maxs = np.max(all_xyz, axis=0)
    length, width, height = np.subtract(maxs, mins) + 3
    array = np.zeros((height, width, length), dtype=np.uint16)

    # Fill voxel array.
    all_xyz_offset = [(z - mins[2] + 1, y - mins[1] + 1, x - mins[0] + 1) for x, y, z in all_xyz]
    array[tuple(np.transpose(all_xyz_offset))] = all_ids

    return array
