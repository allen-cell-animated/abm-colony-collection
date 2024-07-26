import unittest
from math import comb

import networkx as nx
import pandas as pd

from abm_colony_collection.calculate_centrality_measures import calculate_centrality_measures


class TestCalculateCentralityMeasures(unittest.TestCase):
    def test_calculate_centrality_measures(self):
        network = nx.Graph()

        nodes = [1, 2, 3, 4]
        edges = [(1, 2), (1, 3), (3, 4), (1, 4)]
        network.add_nodes_from(nodes)
        network.add_edges_from(edges)

        n = len(nodes) - 1

        expected_measures = pd.DataFrame(
            [
                {
                    "ID": 1,
                    "DEGREE_CENTRALITY": 3 / n,
                    "CLOSENESS_CENTRALITY": n / (1 + 1 + 1),
                    "BETWEENNESS_CENTRALITY": 2 / comb(n, 2),
                },
                {
                    "ID": 2,
                    "DEGREE_CENTRALITY": 1 / n,
                    "CLOSENESS_CENTRALITY": n / (1 + 2 + 2),
                    "BETWEENNESS_CENTRALITY": 0 / comb(n, 2),
                },
                {
                    "ID": 3,
                    "DEGREE_CENTRALITY": 2 / n,
                    "CLOSENESS_CENTRALITY": n / (1 + 2 + 1),
                    "BETWEENNESS_CENTRALITY": 0 / comb(n, 2),
                },
                {
                    "ID": 4,
                    "DEGREE_CENTRALITY": 2 / n,
                    "CLOSENESS_CENTRALITY": n / (1 + 2 + 1),
                    "BETWEENNESS_CENTRALITY": 0 / comb(n, 2),
                },
            ]
        )

        measures = calculate_centrality_measures(network)

        self.assertTrue(expected_measures.equals(measures))


if __name__ == "__main__":
    unittest.main()
