import unittest

import networkx as nx
import pandas as pd

from abm_colony_collection.convert_to_network import convert_to_network


class TestConvertToNetwork(unittest.TestCase):
    def test_convert_to_network(self):
        neighbors = pd.DataFrame(
            [
                {"ID": 1, "NEIGHBORS": [2, 3, 4]},
                {"ID": 2, "NEIGHBORS": [1]},
                {"ID": 3, "NEIGHBORS": [1, 4]},
                {"ID": 4, "NEIGHBORS": [1, 3]},
            ]
        )

        expected_network = nx.Graph()

        nodes = [1, 2, 3, 4]
        edges = [(1, 2), (1, 3), (3, 4), (1, 4)]
        expected_network.add_nodes_from(nodes)
        expected_network.add_edges_from(edges)

        network = convert_to_network(neighbors)

        self.assertTrue(nx.utils.graphs_equal(expected_network, network))


if __name__ == "__main__":
    unittest.main()
