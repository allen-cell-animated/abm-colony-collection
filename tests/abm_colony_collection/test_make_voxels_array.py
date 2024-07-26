import unittest

import numpy as np

from abm_colony_collection.make_voxels_array import make_voxels_array


class TestMakeVoxelsArray(unittest.TestCase):
    def test_make_voxels_array_no_locations(self):
        locations = [
            {
                "id": 1,
                "location": [
                    {
                        "region": "A",
                        "voxels": [],
                    },
                ],
            },
            {
                "id": 2,
                "location": [
                    {
                        "region": "A",
                        "voxels": [],
                    },
                ],
            },
        ]

        expected_array = np.zeros((1, 1, 1))

        array = make_voxels_array(locations)

        self.assertTrue((expected_array == array).all())

    def test_make_voxels_array(self):
        locations = [
            {
                "id": 1,
                "location": [
                    {
                        "region": "A",
                        "voxels": [
                            [1, 1, 1],
                            [1, 2, 1],
                            [1, 1, 2],
                        ],
                    },
                    {
                        "region": "B",
                        "voxels": [
                            [1, 2, 2],
                        ],
                    },
                ],
            },
            {
                "id": 2,
                "location": [
                    {
                        "region": "A",
                        "voxels": [
                            [2, 5, 2],
                            [2, 5, 3],
                            [2, 6, 2],
                            [2, 6, 3],
                        ],
                    },
                ],
            },
        ]

        expected_array = np.zeros((5, 8, 4))
        expected_array[1:3, 1:3, 1] = 1
        expected_array[2:4, 5:7, 2] = 2

        array = make_voxels_array(locations)

        self.assertTrue((expected_array == array).all())


if __name__ == "__main__":
    unittest.main()
