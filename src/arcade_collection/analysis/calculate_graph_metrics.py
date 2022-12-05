from typing import Any, Optional
from dataclasses import dataclass

from prefect import task
import igraph as ig
import numpy as np


@dataclass
class GraphMetrics:
    """
    Dataclass for storing graph metrics
    """

    nodes: int
    edges: int
    radius: float
    diameter: float
    avg_eccentricity: float
    avg_shortest_path: float
    avg_in_degrees: float
    avg_out_degrees: float
    avg_degree: float
    avg_clustering: float
    avg_closeness: float
    avg_betweenness: float
    components: int
    name: Optional[str] = None

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


@task
def calculate_graph_metrics(edges: list[list[int]], weights: list[float] = None) -> GraphMetrics:
    """
    Calculate graph metrics from a set of edges

    Parameters
    ----------
    edges
        List of edges defined by end nodes (i.e. [[1,2], [2,4]...]) of the graph to be analyzed
    weights
        (Optional) List of weights in the same order as the list of edges

    Returns
    -------
    metrics
        GraphMetrics object that holds graph metrics
    """

    if len(edges) == 0:
        raise ValueError("Passed edges do not create valid graph.")
    if not weights:
        weights = [1.0] * len(edges)

    igraph = _make_igraph(edges, weights)

    connected: bool = igraph.is_connected("weak")

    m_dict = {}

    m_dict["nodes"] = igraph.vcount()
    m_dict["edges"] = igraph.ecount()
    ecc_metrics = _calc_eccentricity_metrics(igraph, connected)
    m_dict["radius"], m_dict["diameter"], m_dict["avg_eccentricity"] = ecc_metrics
    m_dict["avg_shortest_path"] = _calc_path_metrics(igraph, connected)
    degree_metrics = _calc_degree_metrics(igraph)
    m_dict["avg_in_degrees"], m_dict["avg_out_degrees"], m_dict["avg_degree"] = degree_metrics
    m_dict["avg_clustering"] = _calc_clustering_metric(igraph)
    m_dict["avg_closeness"] = _calc_closeness_metric(igraph)
    m_dict["avg_betweenness"] = _calc_betweenness_metric(igraph)
    m_dict["components"] = _calc_n_components(igraph, connected)

    metrics = GraphMetrics(**m_dict)

    return metrics


def _calc_eccentricity_metrics(graph: ig.Graph, connected: bool) -> tuple[float, float, float]:
    """Helper function to calculate the radius, diameter, and mean eccentricity from igraph"""
    if not connected:
        return float("inf"), float("inf"), float("inf")

    eccs = graph.eccentricity(mode="all")
    radius = min(eccs)
    diameter = max(eccs)
    average_ecc = np.mean(eccs)
    return radius, diameter, average_ecc


def _calc_path_metrics(graph: ig.Graph, connected: bool) -> float:
    """Helper function to calculate the average shortest length from igraph"""
    if not connected:
        return float("inf")

    norm_factor = len(graph.vs) * (len(graph.vs) - 1)
    total_distance: int = 0

    for node in graph.vs:
        distance = graph.distances(source=node, mode="all")
        total_distance += np.sum(distance)
    return total_distance / norm_factor


def _calc_closeness_metric(graph: ig.Graph) -> float:
    """Helperfunction to calculate average closeness from igraph"""
    closeness = np.array(graph.closeness())
    return np.mean(closeness)


def _calc_betweenness_metric(graph: ig.Graph) -> float:
    """Helper function to calcuate normalized betweenness from igraph"""
    n_nodes = graph.vcount()
    betweenness_norm_factor = (n_nodes - 1) * (n_nodes - 2)
    betweenness = np.array(graph.betweenness()) / betweenness_norm_factor
    return np.mean(betweenness)


def _calc_degree_metrics(graph: ig.Graph) -> tuple[float, ...]:
    """
    Helper function to calculate the average degree (number of edges per node) from igraph

    Parameters
    ----------
    graph: igraph Graph

    Returns
    -------
    return_tuple: tuple containing (average_indegree, average_outdegree, average_degree)
    """

    modes = ("in", "out", "all")
    return_tuple = tuple(np.mean(graph.degree(mode=mode)) for mode in modes)
    return return_tuple


def _calc_clustering_metric(graph: ig.Graph) -> float:
    """
    Helper function to calculate the global transitivity,
    i.e. the Clustering coefficient from igraph
    """

    return graph.transitivity_undirected()


def _calc_n_components(graph: ig.Graph, connected: bool) -> int:
    """Helper function to get the number of components (subgraphs) from igraph."""
    if connected:
        return 1

    return len(graph.decompose(mode="weak"))


def _make_igraph(edges: list[list[int]], weights: list[float]) -> ig.Graph:
    """
    Creates an igraph graph from provided edges for certain graph metric calculations and returns
    a directed version of the graph

    Parameters
    ----------
    edges
        List of edges defined by end nodes (i.e. [[1,2], [2,4]...]) of the graph to be analyzed
    weights
        (Optional) List of weights in the same order as the list of edges

    Returns
    -------
    dir_graph
        Directed igraph
    """

    dir_graph = ig.Graph.TupleList(edges=edges, weights=weights, directed=True)

    return dir_graph
