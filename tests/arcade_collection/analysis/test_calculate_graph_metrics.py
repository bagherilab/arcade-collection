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
            "coreness": 1.42857,
            "components": 1,
        }

        returned_metrics = calculate_graph_metrics.fn(edges)

        for key in expected_metrics:
            with self.subTest(key=key):
                self.assertAlmostEqual(expected_metrics[key], returned_metrics[key], places=5)

    def test_calculate_graph_metrics_given_weighted_edges_calculates_metrics(self) -> None:
        edges = [[1, 3], [2, 3], [3, 4], [4, 5], [5, 6], [5, 7], [6, 7]]
        weights = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        expected_metrics = {
            "nodes": 7,
            "edges": 7,
            "radius": 4.0,
            "diameter": 8.0,
            "avg_eccentricity": 6.85714,
            "avg_shortest_path": 4.47619,
            "avg_in_degrees": 1.0,
            "avg_out_degrees": 1.0,
            "avg_degree": 2.0,
            "avg_clustering": 0.33333,
            "avg_closeness": 0.23149,
            "avg_betweenness": 0.11905,
            "coreness": 1.42857,
            "components": 1,
        }

        returned_metrics = calculate_graph_metrics.fn(edges, weights)

        for key in expected_metrics:
            with self.subTest(key=key):
                self.assertAlmostEqual(expected_metrics[key], returned_metrics[key], places=5)

    def test_calculate_graph_metrics_given_weighted_edges_loop_calculates_metrics(self) -> None:
        edges = [[1, 2], [2, 3], [1, 4], [3, 5], [4, 5]]
        weights = [1.0, 1.0, 1.0, 1.0, 6.0]
        expected_metrics = {
            "nodes": 5,
            "edges": 5,
            "radius": 2.0,
            "diameter": 4.0,
            "avg_eccentricity": 3.2,
            "avg_shortest_path": 2,
            "avg_in_degrees": 1.0,
            "avg_out_degrees": 1.0,
            "avg_degree": 2.0,
            "avg_clustering": 0.0,
            "avg_closeness": 0.52190,
            "avg_betweenness": 0.06667,
            "coreness": 2.0,
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
            "avg_closeness": float("inf"),
            "avg_betweenness": float("inf"),
            "coreness": float("inf"),
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
