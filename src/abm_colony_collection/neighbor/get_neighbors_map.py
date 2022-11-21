from typing import Optional

from prefect import task
import numpy as np
from skimage import measure
from scipy import ndimage


@task
def get_neighbors_map(array: np.ndarray) -> dict:
    neighbors_map: dict = {cell_id: {} for cell_id in np.unique(array)}
    neighbors_map.pop(0, None)

    # Create binary mask for array.
    mask = np.zeros(array.shape, dtype="int")
    mask[array != 0] = 1

    # Label connected groups.
    labels, groups = measure.label(mask, connectivity=2, return_num=True)

    # In line function that returns a filter lambda for a given id
    voxel_filter = lambda voxel_id: lambda v: voxel_id in v

    for group in range(1, groups + 1):
        group_crop = get_cropped_array(array, group, labels)
        voxel_ids = [i for i in np.unique(group_crop) if i != 0]

        # Find neighbors for each voxel id.
        for voxel_id in voxel_ids:
            voxel_crop = get_cropped_array(group_crop, voxel_id, crop_original=True)

            # Apply custom filter to get border locations.
            border_mask = ndimage.generic_filter(voxel_crop, voxel_filter(voxel_id), size=3)

            # Find neighbors overlapping border.
            neighbor_list = np.unique(voxel_crop[border_mask == 1])
            neighbor_list = [i for i in neighbor_list if i not in [0, voxel_id]]
            neighbors_map[voxel_id] = {"group": group, "neighbors": neighbor_list}

    return neighbors_map


def get_bounding_box(array: np.ndarray) -> tuple[int, int, int, int, int, int]:
    """Finds bounding box around binary array."""
    x, y, z = array.shape

    xbounds = np.any(array, axis=(1, 2))
    ybounds = np.any(array, axis=(0, 2))
    zbounds = np.any(array, axis=(0, 1))

    xmin, xmax = np.where(xbounds)[0][[0, -1]]
    ymin, ymax = np.where(ybounds)[0][[0, -1]]
    zmin, zmax = np.where(zbounds)[0][[0, -1]]

    xmin = max(xmin - 1, 0)
    xmax = min(xmax + 2, x)

    ymin = max(ymin - 1, 0)
    ymax = min(ymax + 2, y)

    zmin = max(zmin - 1, 0)
    zmax = min(zmax + 2, z)

    return xmin, xmax, ymin, ymax, zmin, zmax


def get_cropped_array(
    array: np.ndarray, label: int, labels: Optional[np.ndarray] = None, crop_original: bool = False
) -> np.ndarray:
    # Set all voxels not matching label to zero.
    array_mask = array.copy()
    array_filter = labels if labels is not None else array_mask
    array_mask[array_filter != label] = 0

    # Crop array to label.
    xmin, xmax, ymin, ymax, zmin, zmax = get_bounding_box(array_mask)

    if crop_original:
        return array[xmin:xmax, ymin:ymax, zmin:zmax]

    return array_mask[xmin:xmax, ymin:ymax, zmin:zmax]
