from typing import Callable, Optional

import numpy as np
from scipy import ndimage
from skimage import measure


def get_neighbors_map(array: np.ndarray) -> dict:
    """
    Creates map of region ids to lists of neighbors.

    Each region id is also assigned a group number, where all regions in a given
    group are simply connected.

    Parameters
    ----------
    array
        Segmentation array.

    Returns
    -------
    :
        Map of id to group and neighbor ids.
    """

    neighbors_map: dict = {cell_id: {} for cell_id in np.unique(array)}
    neighbors_map.pop(0, None)

    # Create binary mask for array.
    mask = np.zeros(array.shape, dtype="int")
    mask[array != 0] = 1

    # Label connected groups.
    labels, groups = measure.label(mask, connectivity=2, return_num=True)

    for group in range(1, groups + 1):
        group_crop = get_cropped_array(array, group, labels)
        voxel_ids = [i for i in np.unique(group_crop) if i != 0]

        # Find neighbors for each voxel id.
        for voxel_id in voxel_ids:
            voxel_crop = get_cropped_array(group_crop, voxel_id, crop_original=True)

            # Apply custom filter to get border locations.
            border_mask = ndimage.generic_filter(voxel_crop, _get_voxel_id_filter(voxel_id), size=3)

            # Find neighbors overlapping border.
            neighbor_list = np.unique(voxel_crop[border_mask == 1])
            neighbor_list = [i for i in neighbor_list if i not in [0, voxel_id]]
            neighbors_map[voxel_id] = {"group": group, "neighbors": neighbor_list}

    return neighbors_map


def _get_voxel_id_filter(voxel_id: int) -> Callable:
    """Create filtering lambda for given id."""
    return lambda v: voxel_id in v


def get_bounding_box(array: np.ndarray) -> tuple[int, int, int, int, int, int]:
    """
    Find bounding box around array.

    Bounds are calculated with a one-voxel border, if possible.

    Parameters
    ----------
    array
        Segmentation array.

    Returns
    -------
    :
        The bounding box (xmin, xmax, ymin, ymax, zmin, zmax) indices
    """

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
    """
    Crop array around label region.

    Parameters
    ----------
    array
        Array to crop.
    label
        Region label.
    labels
        Array of all region labels.
    crop_original
        True to crop the original array keeping all labels, False otherwise.

    Returns
    -------
    :
        Cropped array.
    """

    # Set all voxels not matching label to zero.
    array_mask = array.copy()
    array_filter = labels if labels is not None else array_mask
    array_mask[array_filter != label] = 0

    # Crop array to label.
    xmin, xmax, ymin, ymax, zmin, zmax = get_bounding_box(array_mask)

    if crop_original:
        return array[xmin:xmax, ymin:ymax, zmin:zmax]

    return array_mask[xmin:xmax, ymin:ymax, zmin:zmax]
