"""Rewire a DAG while preserving every node's in/out degree and task ancestors."""

import argparse
import json
import os
import pickle
import random
from pathlib import Path

import networkx as nx
import pandas as pd


def perturb_graph(input_path: Path, output_path: Path, seed: int, target_node: str):
    with input_path.open("rb") as stream:
        graph = pickle.load(stream)
    if not isinstance(graph, pd.DataFrame) or graph.shape[0] != graph.shape[1] or list(graph.index) != list(graph.columns):
        raise ValueError("Expected a square pandas adjacency matrix with matching labels")
    values = graph.to_numpy()
    if not ((values == 0) | (values == 1)).all() or values.diagonal().any():
        raise ValueError("Expected a loop-free binary adjacency matrix")

    nodes = list(graph.index)
    if target_node not in nodes:
        raise ValueError(f"Target {target_node!r} not found in graph")
    original = nx.DiGraph()
    original.add_nodes_from(nodes)
    original.add_edges_from((nodes[i], nodes[j]) for i, j in zip(*values.nonzero()))
    if not nx.is_directed_acyclic_graph(original):
        raise ValueError("Input graph must be a DAG")
    if original.out_degree(target_node) != 0:
        raise ValueError("Task node must be a sink")

    original_edges = set(original.edges())
    original_ancestors = nx.ancestors(original, target_node)
    if len(original_edges) < 2:
        raise ValueError("At least two edges are needed for degree-preserving rewiring")
    rng = random.Random(seed)
    rewired = original.copy()

    # A directed 2-switch preserves each node's in-degree and out-degree.
    # Accept only DAGs with the same exact task-ancestor set, so the C2BM
    # propagator inventory and every per-node equation shape remain compatible.
    successful = 0
    attempts = max(10000, len(original_edges) * 1000)
    for _ in range(attempts):
        edges = list(rewired.edges())
        (a, b), (c, d) = rng.sample(edges, 2)
        if a == c or b == d or a == d or c == b:
            continue
        replacement = {(a, d), (c, b)}
        if any(rewired.has_edge(u, v) for u, v in replacement):
            continue
        rewired.remove_edge(a, b)
        rewired.remove_edge(c, d)
        rewired.add_edges_from(replacement)
        valid = nx.is_directed_acyclic_graph(rewired)
        if valid:
            valid = nx.ancestors(rewired, target_node) == original_ancestors
        if valid:
            successful += 1
        else:
            rewired.remove_edges_from(replacement)
            rewired.add_edge(a, b)
            rewired.add_edge(c, d)
        if successful >= max(5, len(original_edges)) and set(rewired.edges()) != original_edges:
            break

    if set(rewired.edges()) == original_edges:
        raise RuntimeError("Could not find a distinct DAG with preserved degrees and task ancestors")
    if any(original.in_degree(n) != rewired.in_degree(n) or original.out_degree(n) != rewired.out_degree(n) for n in nodes):
        raise RuntimeError("Internal error: degree sequence changed")

    result = pd.DataFrame(0, index=nodes, columns=nodes, dtype=int)
    for source, destination in rewired.edges():
        result.loc[source, destination] = 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_graph = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    with temporary_graph.open("wb") as stream:
        pickle.dump(result, stream)
    os.replace(temporary_graph, output_path)
    shared = len(original_edges & set(rewired.edges()))
    metadata = {
        "input_graph": str(input_path), "output_graph": str(output_path), "seed": seed,
        "target_node": target_node, "node_count": len(nodes), "edge_count": len(original_edges),
        "shared_directed_edges": shared, "removed_original_edges": len(original_edges) - shared,
        "added_new_edges": len(original_edges) - shared,
        "edge_overlap_fraction": shared / len(original_edges),
        "in_degree_preserved_per_node": True, "out_degree_preserved_per_node": True,
        "task_ancestor_set_preserved": True, "successful_degree_preserving_switches": successful,
    }
    metadata_path = output_path.with_suffix(".json")
    temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.tmp")
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_metadata, metadata_path)
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--target-node", required=True)
    args = parser.parse_args()
    perturb_graph(args.input, args.output, args.seed, args.target_node)


if __name__ == "__main__":
    main()
