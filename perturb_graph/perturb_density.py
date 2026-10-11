"""Generate seeded random DAGs at distinct densities around a reference graph."""

import argparse
import json
import math
import pickle
import random
from pathlib import Path

import networkx as nx
import pandas as pd


def create_density_graph(
    input_path: Path,
    output_path: Path,
    graph_seed: int,
    target_node: str | None,
    min_dag_density: float,
    max_dag_density: float,
    density_seed: int,
    n_graphs: int,
):
    with input_path.open("rb") as stream:
        original = pickle.load(stream)
    if not isinstance(original, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame adjacency matrix, got {type(original).__name__}")
    if original.shape[0] != original.shape[1] or list(original.index) != list(original.columns):
        raise ValueError("Graph must be square with matching row and column labels in the same order")
    values = original.to_numpy()
    if not (((values == 0) | (values == 1)).all()) or values.diagonal().any():
        raise ValueError("Expected a binary adjacency matrix without self-loops")

    nodes = list(original.index)
    target_node = target_node or nodes[-1]
    if target_node not in nodes:
        raise ValueError(f"Target {target_node!r} is not a graph node")
    n = len(nodes)
    max_dag_edges = n * (n - 1) // 2
    min_edges = max(1, math.ceil(min_dag_density * max_dag_edges))
    max_edges = min(max_dag_edges, math.floor(max_dag_density * max_dag_edges))
    if not 0 <= min_dag_density <= max_dag_density <= 1:
        raise ValueError("Require 0 <= min-dag-density <= max-dag-density <= 1")
    if max_edges - min_edges + 1 < n_graphs:
        raise ValueError("Density interval must contain at least n-graphs distinct edge counts")
    if not 1 <= graph_seed <= n_graphs:
        raise ValueError(f"graph-seed must be in 1..{n_graphs}")

    # Select distinct edge counts once for the experiment; each Condor process
    # derives the same list and takes the value associated with its graph seed.
    count_rng = random.Random(density_seed)
    selected_edge_counts = count_rng.sample(range(min_edges, max_edges + 1), n_graphs)
    edge_count = selected_edge_counts[graph_seed - 1]

    original_edges = {
        (nodes[i], nodes[j])
        for i in range(n) for j in range(n) if values[i, j] == 1
    }
    original_dag = nx.DiGraph()
    original_dag.add_nodes_from(nodes)
    original_dag.add_edges_from(original_edges)
    if not nx.is_directed_acyclic_graph(original_dag):
        raise ValueError("Reference graph must be a DAG")
    if original_dag.out_degree(target_node) != 0:
        raise ValueError(f"Target {target_node!r} must be a sink")

    rng = random.Random(graph_seed)
    chosen_edges = None
    for _ in range(1000):
        order = [node for node in nodes if node != target_node]
        rng.shuffle(order)
        order.append(target_node)
        candidates = [(order[i], order[j]) for i in range(n) for j in range(i + 1, n)]
        target_parent_edges = [edge for edge in candidates if edge[1] == target_node]
        mandatory = rng.choice(target_parent_edges)
        remaining = [edge for edge in candidates if edge != mandatory]
        selected = {mandatory, *rng.sample(remaining, edge_count - 1)}
        if selected != original_edges:
            chosen_edges = selected
            break
    if chosen_edges is None:
        raise RuntimeError("Could not produce a graph distinct from the reference graph")

    perturbed = pd.DataFrame(0, index=nodes, columns=nodes, dtype=int)
    for source, destination in chosen_edges:
        perturbed.loc[source, destination] = 1
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(chosen_edges)
    if not nx.is_directed_acyclic_graph(graph) or graph.out_degree(target_node) != 0:
        raise RuntimeError("Generated graph failed DAG or target-sink validation")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        pickle.dump(perturbed, stream)

    original_edge_count = len(original_edges)
    shared_edges = len(original_edges & chosen_edges)
    metadata = {
        "input_graph": str(input_path),
        "output_graph": str(output_path),
        "graph_seed": graph_seed,
        "density_seed": density_seed,
        "density_seed_edge_counts": selected_edge_counts,
        "target_node": target_node,
        "node_count": n,
        "max_dag_edges": max_dag_edges,
        "edge_count_original": original_edge_count,
        "edge_count_perturbed": len(chosen_edges),
        "reference_dag_density": original_edge_count / max_dag_edges,
        "perturbed_dag_density": len(chosen_edges) / max_dag_edges,
        "perturbed_directed_density": len(chosen_edges) / (n * (n - 1)),
        "min_dag_density": min_dag_density,
        "max_dag_density": max_dag_density,
        "shared_directed_edges": shared_edges,
        "removed_original_edges": original_edge_count - shared_edges,
        "added_new_edges": len(chosen_edges) - shared_edges,
        "adjacency_hamming_distance": int((original.to_numpy() != perturbed.to_numpy()).sum()),
        "is_dag": True,
        "target_is_sink": True,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph-seed", type=int, required=True)
    parser.add_argument("--target-node", default=None)
    parser.add_argument("--min-dag-density", type=float, default=0.30)
    parser.add_argument("--max-dag-density", type=float, default=0.80)
    parser.add_argument("--density-seed", type=int, default=31415)
    parser.add_argument("--n-graphs", type=int, default=5)
    args = parser.parse_args()
    create_density_graph(
        args.input, args.output, args.graph_seed, args.target_node,
        args.min_dag_density, args.max_dag_density, args.density_seed, args.n_graphs,
    )


if __name__ == "__main__":
    main()
