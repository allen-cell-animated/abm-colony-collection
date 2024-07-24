import unittest

import networkx as nx
import pandas as pd

from abm_colony_collection.calculate_distance_measures import calculate_distance_measures


class TestCalculateDistanceMeasures(unittest.TestCase):
    def test_calculate_distance_measures(self):
        network = nx.Graph()

        nodes = [1, 2, 3, 4]
        edges = [(1, 2), (1, 3), (3, 4), (1, 4)]
        network.add_nodes_from(nodes)
        network.add_edges_from(edges)

        expected_measures = pd.DataFrame(
            [
                {
                    "ID": 1,
                    "ECCENTRICITY": 1,
                },
                {
                    "ID": 2,
                    "ECCENTRICITY": 2,
                },
                {
                    "ID": 3,
                    "ECCENTRICITY": 2,
                },
                {
                    "ID": 4,
                    "ECCENTRICITY": 2,
                },
            ]
        )

        measures = calculate_distance_measures(network)

        self.assertTrue(expected_measures.equals(measures))


if __name__ == "__main__":
    unittest.main()
