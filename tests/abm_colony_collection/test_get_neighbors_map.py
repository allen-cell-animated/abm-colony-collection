import unittest

import numpy as np

from abm_colony_collection.get_neighbors_map import (
    get_bounding_box,
    get_cropped_array,
    get_neighbors_map,
)


class TestGetNeighborsMap(unittest.TestCase):
    def setUp(self) -> None:
        array = np.zeros((10, 10, 10))

        # Layer does not extend close to edge, so layer is not used to define
        # the edge cell ids.
        array[1, 2:5, 2:5] = 1
        array[1, 2:5, 5:8] = 2
        array[1, 5:8, 2:5] = 3
        array[1, 5:8, 5:8] = 4

        # Layer extends closer to edge, so layer should be selected as slice.
        array[3, 1:5, 1:5] = 5
        array[3, 1:5, 5:9] = 6
        array[3, 5:9, 1:5] = 7
        array[3, 5:9, 5:9] = 8

        # Center of selected layer should not be included in edge ids.
        array[1, 3:7, 3:7] = 9
        array[3, 3:7, 3:7] = 10

        # Add hole to check dilation.
        array[3, 4:5, 4:6] = 0

        self.array = array

    def test_get_neighbors_map(self):

        expected_neighbors_map = {
            1: {"group": 1, "neighbors": [2, 3, 9]},
            2: {"group": 1, "neighbors": [1, 4, 9]},
            3: {"group": 1, "neighbors": [1, 4, 9]},
            4: {"group": 1, "neighbors": [2, 3, 9]},
            5: {"group": 2, "neighbors": [6, 7, 10]},
            6: {"group": 2, "neighbors": [5, 8, 10]},
            7: {"group": 2, "neighbors": [5, 8, 10]},
            8: {"group": 2, "neighbors": [6, 7, 10]},
            9: {"group": 1, "neighbors": [1, 2, 3, 4]},
            10: {"group": 2, "neighbors": [5, 6, 7, 8]},
        }

        neighbors_map = get_neighbors_map(self.array)

        self.assertDictEqual(expected_neighbors_map, neighbors_map)

    def test_get_cropped_array_no_labels_no_crop_original(self):
        label = 1

        expected_cropped = np.zeros((3, 5, 5))
        expected_cropped[1, 1, 1:4] = 1
        expected_cropped[1, 1:4, 1] = 1

        cropped = get_cropped_array(self.array, label, labels=None, crop_original=False)

        self.assertTrue((expected_cropped == cropped).all())

    def test_get_cropped_array_with_labels_no_crop_original(self):
        label = 1

        labels = np.zeros((10, 10, 10))
        labels[1, 1:4, 1:4] = 1

        expected_cropped = np.zeros((3, 4, 4))
        expected_cropped[1, 1:3, 1:3] = 1
        expected_cropped[1, 2, 2] = 9

        cropped = get_cropped_array(self.array, label, labels=labels, crop_original=False)

        self.assertTrue((expected_cropped == cropped).all())

    def test_get_cropped_array_no_labels_crop_original(self):
        label = 1

        expected_cropped = np.zeros((3, 5, 5))
        expected_cropped[1, 1:4, 1:4] = 1
        expected_cropped[1, 2:5, 2:5] = 9
        expected_cropped[1, 1, 4] = 2
        expected_cropped[1, 4, 1] = 3

        array = self.array.copy()
        cropped = get_cropped_array(array, label, labels=None, crop_original=True)

        self.assertTrue((expected_cropped == cropped).all())

    def test_get_cropped_array_with_labels_crop_original(self):
        label = 1

        labels = np.zeros((10, 10, 10))
        labels[1, 1:4, 1:4] = 1

        expected_cropped = np.zeros((3, 4, 4))
        expected_cropped[1, 1:4, 1:4] = 1
        expected_cropped[1, 2:4, 2:4] = 9

        cropped = get_cropped_array(self.array, label, labels=labels, crop_original=True)

        self.assertTrue((expected_cropped == cropped).all())

    def test_get_bounding_box(self):
        parameters = [
            ((2, 4, 2, 4, 2, 4), (1, 5, 1, 5, 1, 5)),  # no adjustments
            ((0, 4, 0, 4, 0, 4), (0, 5, 0, 5, 0, 5)),  # adjust minimums
            ((2, 10, 2, 12, 2, 14), (1, 10, 1, 12, 1, 14)),  # adjust maximums
        ]

        for minsmaxs, expected_bounds in parameters:
            with self.subTest(bounds=expected_bounds):
                zmin, zmax, ymin, ymax, xmin, xmax = minsmaxs
                array = np.zeros((10, 12, 14))
                array[zmin:zmax, ymin:ymax, xmin:xmax] = 1

                bounds = get_bounding_box(array)

                self.assertTupleEqual(expected_bounds, bounds)


if __name__ == "__main__":
    unittest.main()
