import unittest
from dataclasses import asdict

from prefect import flow, task

from arcade_collection.analysis import calculate_graph_metrics


class TestCalculateGraphMetrics(unittest.TestCase):
    def test_calculate_graph_metrics_given_unweighted_edges_calculates_metrics(self):
        edges = [[1, 3], [2, 3], [3, 4], [4, 5], [5, 6], [5, 7], [6, 7]]
        expected_metrics = {
            "nodes": 7,
            "edges": 7,
            "radius": 2.0,
            "diameter": 4.0,
            "avg_eccentricity": 3.42857,
            "avg_shortest_path": 1.07143,
            "avg_in_degrees": 1.0,
            "avg_out_degrees": 1.0,
            "avg_degree": 2.0,
            "avg_clustering": 0.33333,
            "avg_closeness": 0.23246,
            "avg_betweenness": 0.11905,
            "components": 1,
        }

        returned_metrics = calculate_graph_metrics.fn(edges)

        for key in expected_metrics:
            with self.subTest(key=key):
                self.assertAlmostEqual(expected_metrics[key], returned_metrics[key], places=5)

    def test_calculate_graph_metrics_given_weighted_edges_calculates_metrics(self):
        edges = [[1, 3], [2, 3], [3, 4], [4, 5], [5, 6], [5, 7], [6, 7]]
        weights = [2, 1, 3, 3, 1, 5, 1]
        expected_metrics = {
            "nodes": 7,
            "edges": 7,
            "radius": 2.0,
            "diameter": 4.0,
            "avg_eccentricity": 3.42857,
            "avg_shortest_path": 1.07143,
            "avg_in_degrees": 1.0,
            "avg_out_degrees": 1.0,
            "avg_degree": 2.0,
            "avg_clustering": 0.33333,
            "avg_closeness": 0.23246,
            "avg_betweenness": 0.11905,
            "components": 1,
        }

        returned_metrics = calculate_graph_metrics.fn(edges, weights)

        for key in expected_metrics:
            with self.subTest(key=key):
                self.assertAlmostEqual(expected_metrics[key], returned_metrics[key], places=5)

    def test_calculate_graph_metrics_given_disconnected_eges_calculates_metrics(self):
        edges = [[1, 3], [2, 3], [3, 4], [5, 6], [5, 7], [6, 7]]
        expected_metrics = {
            "num_edges": 6,
            "num_nodes": 7,
            "radius": 1.0,
            "diameter": 1.5,
            "avg_in_degrees": 0.85714,
            "avg_out_degrees": 0.85714,
            "avg_degree": 1.71429,
            "avg_ecc": 1.375,
            "path": 0.54167,
            "avg_clust": 0.5,
            "avg_clos": 0.16190,
            "avg_betw": 0.00952,
            "num_comps": 2,
        }

        returned_metrics = calculate_graph_metrics.fn(edges)

        for key in expected_metrics:
            with self.subTest(key=key):
                self.assertAlmostEqual(expected_metrics[key], returned_metrics[key], places=5)

    def test_calculate_graph_metrics_given_empty_edges_returns_none(self):
        edges = []
        expected_metrics = {
            "num_edges": None,
            "num_nodes": None,
            "radius": None,
            "diameter": None,
            "avg_in_degrees": None,
            "avg_out_degrees": None,
            "avg_degree": None,
            "avg_ecc": None,
            "path": None,
            "avg_clust": None,
            "avg_clos": None,
            "avg_betw": None,
            "num_comps": None,
        }

        returned_metrics = calculate_graph_metrics.fn(edges)

        for key in expected_metrics:
            with self.subTest(key=key):
                self.assertAlmostEqual(expected_metrics[key], returned_metrics[key], places=5)


if __name__ == "__main__":
    unittest.main()
