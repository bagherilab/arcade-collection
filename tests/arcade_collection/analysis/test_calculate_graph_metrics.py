import unittest
from dataclasses import asdict

from prefect import flow, task

from arcade_collection.analysis import calculate_graph_metrics


class TestCalculateGraphMetrics(unittest.TestCase):
    def test_calculate_graph_metrics_given_unweighted_edges_calculates_metrics(self) -> None:
        edges = [[1, 3], [2, 3], [3, 4], [4, 5], [5, 6], [5, 7], [6, 7]]
        expected_metrics = {
            "nodes": 7,
            "edges": 7,
            "radius": 2.0,
            "diameter": 4.0,
            "avg_eccentricity": 3.42857,
            "avg_shortest_path": 2.23810,
            "avg_in_degrees": 1.0,
            "avg_out_degrees": 1.0,
            "avg_degree": 2.0,
            "avg_clustering": 0.33333,
            "avg_closeness": 0.46299,
            "avg_betweenness": 0.11905,
            "components": 1,
        }

        returned_metrics = calculate_graph_metrics.fn(edges)

        for key in expected_metrics:
            with self.subTest(key=key):
                self.assertAlmostEqual(expected_metrics[key], returned_metrics[key], places=5)

    def test_calculate_graph_metrics_given_weighted_edges_calculates_metrics(self) -> None:
        edges = [[1, 3], [2, 3], [3, 4], [4, 5], [5, 6], [5, 7], [6, 7]]
        weights = [2.0, 1.0, 3.0, 3.0, 1.0, 5.0, 1.0]
        expected_metrics = {
            "nodes": 7,
            "edges": 7,
            "radius": 2.0,
            "diameter": 4.0,
            "avg_eccentricity": 3.42857,
            "avg_shortest_path": 2.23810,
            "avg_in_degrees": 1.0,
            "avg_out_degrees": 1.0,
            "avg_degree": 2.0,
            "avg_clustering": 0.33333,
            "avg_closeness": 0.46299,
            "avg_betweenness": 0.11905,
            "components": 1,
        }

        returned_metrics = calculate_graph_metrics.fn(edges, weights)

        for key in expected_metrics:
            with self.subTest(key=key):
                self.assertAlmostEqual(expected_metrics[key], returned_metrics[key], places=5)

    def test_calculate_graph_metrics_given_disconnected_eges_calculates_metrics(self) -> None:
        edges = [[1, 3], [2, 3], [3, 4], [5, 6], [5, 7], [6, 7]]
        expected_metrics = {
            "nodes": 7,
            "edges": 6,
            "radius": float("inf"),
            "diameter": float("inf"),
            "avg_eccentricity": float("inf"),
            "avg_shortest_path": float("inf"),
            "avg_in_degrees": 0.85714,
            "avg_out_degrees": 0.85714,
            "avg_degree": 1.71429,
            "avg_clustering": 0.5,
            "avg_closeness": 0.82857,
            "avg_betweenness": 0.00952,
            "components": 2,
        }

        returned_metrics = calculate_graph_metrics.fn(edges)

        for key in expected_metrics:
            with self.subTest(key=key):
                self.assertAlmostEqual(expected_metrics[key], returned_metrics[key], places=5)

    def test_calculate_graph_metrics_given_empty_edges_returns_none(self) -> None:
        edges: list[list[int]] = []
        with self.assertRaises(ValueError):
            calculate_graph_metrics.fn(edges)


if __name__ == "__main__":
    unittest.main()
