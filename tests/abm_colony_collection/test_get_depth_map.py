import unittest

import numpy as np

from abm_colony_collection.get_depth_map import find_edge_ids, get_depth_map


class TestGetDepthMap(unittest.TestCase):
    def setUp(self):
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

        neighbors_map = {
            1: {"neighbors": [2, 3, 9, 5]},
            2: {"neighbors": [1, 4, 9, 6]},
            3: {"neighbors": [1, 4, 9, 7]},
            4: {"neighbors": [2, 3, 9, 8]},
            5: {"neighbors": [6, 7, 10, 1]},
            6: {"neighbors": [5, 8, 10, 2]},
            7: {"neighbors": [5, 8, 10, 3]},
            8: {"neighbors": [6, 7, 10, 4]},
            9: {"neighbors": [1, 2, 3, 4, 10]},
            10: {"neighbors": [5, 6, 7, 8, 9]},
        }

        self.neighbors_map = neighbors_map

    def test_get_depth_map(self):
        expected_depth_map = {1: 2, 2: 2, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1, 9: 3, 10: 2}

        depth_map = get_depth_map(self.array, self.neighbors_map)
        self.assertDictEqual(expected_depth_map, depth_map)

    def test_get_depth_map_empty_array(self):
        array = np.zeros((1, 1, 1))
        depth_map = get_depth_map(array, {})
        self.assertDictEqual({}, depth_map)

    def test_find_edge_ids(self):
        expected_edge_ids = [5, 6, 7, 8]
        edges_ids = find_edge_ids(self.array)
        self.assertCountEqual(expected_edge_ids, edges_ids)


if __name__ == "__main__":
    unittest.main()
