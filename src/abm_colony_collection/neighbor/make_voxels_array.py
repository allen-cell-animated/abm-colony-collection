from prefect import task
import numpy as np


@task
def make_voxels_array(locations: dict) -> np.ndarray:
    # Extract all voxel positions with id.
    all_ids: list[int] = []
    all_xyz: list[tuple[int, int, int]] = []
    for location in locations:
        cell_id = location["id"]
        xyz = [(x, y, z) for region in location["location"] for x, y, z in region["voxels"]]
        all_xyz = all_xyz + xyz
        all_ids = all_ids + [cell_id] * len(xyz)

    # Create empty array.
    mins = np.min(all_xyz, axis=0)
    maxs = np.max(all_xyz, axis=0)
    length, width, height = np.subtract(maxs, mins) + 3
    array = np.zeros((height, width, length), dtype=np.uint16)

    # Return if no voxels.
    if len(all_ids) == 0:
        return array

    # Fill voxel array.
    all_xyz_offset = [(z - mins[2] + 1, y - mins[1] + 1, x - mins[0] + 1) for x, y, z in all_xyz]
    array[tuple(np.transpose(all_xyz_offset))] = all_ids

    return array
