from typing import Tuple

from prefect import task
import networkx as nx
import igraph as ig
import numpy as np

@task
def calculate_graph_metrics(edges: list[list[int]], weights: list[int]=None) -> dict[str, int]:
    """
    Calculate graph metrics from a set of edges

    Parameters
    ----------
    edges
        List of edges ([1,2], [2,4]...) of the graph to be analyzed
    weights
        (Optional) List of weights in the same order as the list of edges

    Returns
    -------
        : dict
    """
    dir_nxgraph, undir_nxgraph = _make_nxgraphs(edges, weights)
    dir_igraph, undir_igraph = _make_igraphs(edges, weights)

    metrics_dict = {"num_edges": None,
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
                    "num_comps": None, }

    if not dir_nxgraph:
        return metrics_dict

    metrics_dict["num_edges"] = dir_nxgraph.number_of_edges()
    metrics_dict["num_nodes"] = dir_nxgraph.number_of_nodes()

    # Normalization factor for betweenness that igraph does not automatically use
    betweenness_norm_factor = (metrics_dict["num_nodes"] - 1) * (metrics_dict["num_nodes"] - 2)

    if not nx.is_connected(undir_nxgraph):
        undir_icomponents = undir_igraph.decompose()
        dir_icomponents = dir_igraph.decompose()

        undir_nxcomponents = [undir_nxgraph.subgraph(h) for h in nx.connected_components(undir_nxgraph)]
        dir_nxcomponents = [dir_nxgraph.subgraph(g) for g in nx.connected_components(undir_nxgraph)]

        eccs = [h.eccentricity() for h in undir_icomponents]
        closeness = list(nx.closeness_centrality(dir_nxgraph).values())

        betweenness = dir_igraph.betweenness()
        betweenness = [betw / betweenness_norm_factor for betw in betweenness]

        radii = [min(ecc) for ecc in eccs]
        diameters = [max(ecc) for ecc in eccs]
        paths = [nx.average_shortest_path_length(g) for g in dir_nxcomponents]

        metrics_dict["radius"] = round(np.mean(radii), 5)
        metrics_dict["diameter"] = round(np.mean(diameters), 5)
        metrics_dict["avg_ecc"] = round(np.mean([np.mean(ecc) for ecc in eccs]), 5)
        metrics_dict["path"] = round(np.mean(paths), 5)
        metrics_dict["avg_in_degrees"] = round(np.mean(dir_igraph.indegree()), 5)
        metrics_dict["avg_out_degrees"] = round(np.mean(dir_igraph.outdegree()), 5)
        metrics_dict["avg_degree"] = round(np.mean(undir_igraph.degree()), 5)

        metrics_dict["avg_clust"] = round(undir_igraph.transitivity_undirected(), 5)
        metrics_dict["avg_clos"] = round(np.mean(closeness), 5)
        metrics_dict["avg_betw"] = round(np.mean(betweenness), 5)
        metrics_dict["num_comps"] = len(undir_icomponents)
    else:
        ecc = undir_igraph.eccentricity()
        closeness = list(nx.closeness_centrality(dir_nxgraph).values())
        betweenness = dir_igraph.betweenness()
        betweenness = [betw / betweenness_norm_factor for betw in betweenness]
        metrics_dict["radius"] = round(min(ecc), 5)
        metrics_dict["diameter"] = round(max(ecc), 5)
        metrics_dict["path"] = round(nx.average_shortest_path_length(dir_nxgraph), 5)
        metrics_dict["avg_in_degrees"] = round(np.mean(dir_igraph.indegree()), 5)
        metrics_dict["avg_out_degrees"] = round(np.mean(dir_igraph.outdegree()), 5)
        metrics_dict["avg_degree"] = round(np.mean(undir_igraph.degree()), 5)
        metrics_dict["avg_ecc"] = round(np.mean(ecc), 5)
        metrics_dict["avg_clust"] = round(undir_igraph.transitivity_undirected(), 5)
        metrics_dict["avg_clos"] = round(np.mean(closeness), 5)
        metrics_dict["avg_betw"] = round(np.mean(betweenness), 5)
        metrics_dict["num_comps"] = 1

    return metrics_dict

def _make_nxgraphs(edges: list[list[int]], weights: list[int]) -> Tuple[nx.Graph, nx.Graph]:
    """
    Creates a networkx graph from provided edges for certain graph metric calculations and returns
    a tuple with a directed and undirected version of the graph

    Parameters
    ----------
    edges
        List of edges ([1,2], [2,4]...) of the graph to be analyzed
    weights
        (Optional) List of weights in the same order as the list of edges

    Returns
    -------
        : Tuple[nx.Graph, nx.Graph]
    """
    if not weights:
        weights = [1] * len(edges)

    if len(edges) == 0:
        return None

    dir_graph = nx.DiGraph()
    for idx, edge in enumerate(edges):
        dir_graph.add_edge(edge[0], edge[1], weight=weights[idx])
    undir_graph = nx.Graph(dir_graph)

    return (dir_graph, undir_graph)

def _make_igraphs(edges: list[list[int]], weights: list[int]) -> Tuple[ig.Graph, ig.Graph]:
    """
    Creates an igraph graph from provided edges for certain graph metric calculations and returns
    a tuple with a directed and undirected version of the graph

    Parameters
    ----------
    edges
        List of edges ([1,2], [2,4]...) of the graph to be analyzed
    weights
        (Optional) List of weights in the same order as the list of edges

    Returns
    -------
        : Tuple[nx.Graph, nx.Graph]
    """
    if not weights:
        weights = [1] * len(edges)

    vertices = set()
    edges_augmented = []
    for idx, edge in enumerate(edges):
        vertices.update(edge)
        weighted_edge = [edge[0], edge[1], weights[idx]]
        edges_augmented.append(weighted_edge)
    vertices = sorted(vertices)

    if len(vertices) == 0:
        return None

    dir_graph = ig.Graph.TupleList(edges=edges_augmented, directed=True)
    undir_graph = dir_graph.as_undirected()

    return (dir_graph, undir_graph)
