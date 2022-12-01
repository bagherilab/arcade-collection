import unittest
from prefect import flow, task

from arcade_collection.analysis import calculate_graph_metrics


class TestCalculateGraphMetrics(unittest.TestCase):
    def test_calculate_graph_metrics_given_unweighted_edges_calculates_metrics(self):
        edges = [[1, 3], [2, 3], [3, 4], [4, 5], [5, 6], [5, 7], [6, 7]]
        expected_metrics = {
            "num_edges": 7,
            "num_nodes": 7,
            "radius": 2.0,
            "diameter": 4.0,
            "avg_in_degrees": 1.0,
            "avg_out_degrees": 1.0,
            "avg_degree": 2.0,
            "avg_ecc": 3.42857,
            "path": 1.07143,
            "avg_clust": 0.33333,
            "avg_clos": 0.23246,
            "avg_betw": 0.11905,
            "num_comps": 1,
        }

        returned_metrics = calculate_graph_metrics.fn(edges)
        self.assertDictEqual(expected_metrics, returned_metrics)

    def test_calculate_graph_metrics_given_weighted_edges_calculates_metrics(self):
        edges = [[1, 3], [2, 3], [3, 4], [4, 5], [5, 6], [5, 7], [6, 7]]
        weights = [2, 1, 3, 3, 1, 5, 1]
        expected_metrics = {
            "num_edges": 7,
            "num_nodes": 7,
            "radius": 2.0,
            "diameter": 4.0,
            "avg_in_degrees": 1.0,
            "avg_out_degrees": 1.0,
            "avg_degree": 2.0,
            "avg_ecc": 3.42857,
            "path": 1.07143,
            "avg_clust": 0.33333,
            "avg_clos": 0.23246,
            "avg_betw": 0.11905,
            "num_comps": 1,
        }

        returned_metrics = calculate_graph_metrics.fn(edges, weights)
        self.assertDictEqual(expected_metrics, returned_metrics)


if __name__ == "__main__":
    unittest.main()
