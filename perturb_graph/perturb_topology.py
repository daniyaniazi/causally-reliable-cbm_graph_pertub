"""Create a random DAG topology with the same nodes and edge count as a saved graph."""

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd


def perturb_graph(input_path: Path, output_path: Path, seed: int, target_node: Optional[str]):
    with input_path.open("rb") as stream:
        graph = pickle.load(stream)

    if not isinstance(graph, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame adjacency matrix, got {type(graph).__name__}")
    if graph.shape[0] != graph.shape[1] or list(graph.index) != list(graph.columns):
        raise ValueError("Graph must be square with matching row and column labels in the same order")

    values = graph.to_numpy()
    if not ((values == 0) | (values == 1)).all():
        raise ValueError("Expected a fully directed 0/1 adjacency matrix")
    if values.diagonal().any():
        raise ValueError("Self-loops are not supported")

    nodes = list(graph.index)
    target_node = target_node or nodes[-1]
    if target_node not in nodes:
        raise ValueError(f"Target node {target_node!r} is not in graph labels: {nodes}")

    original_edges = {
        (nodes[i], nodes[j])
        for i in range(len(nodes))
        for j in range(len(nodes))
        if values[i, j] == 1
    }
    original_dag = nx.DiGraph()
    original_dag.add_nodes_from(nodes)
    original_dag.add_edges_from(original_edges)
    if not nx.is_directed_acyclic_graph(original_dag):
        raise ValueError("Input graph must be a DAG before topology perturbation")
    if original_dag.out_degree(target_node) != 0:
        raise ValueError(f"Target {target_node!r} must be a sink (have no outgoing edges)")

    edge_count = len(original_edges)
    if edge_count == 0:
        raise ValueError("Cannot perturb a graph with zero edges")
    if len(nodes) < 2:
        raise ValueError("Need at least two graph nodes")

    rng = random.Random(seed)
    other_nodes = [node for node in nodes if node != target_node]
    # The task node remains last, matching the project convention that it is a sink.
    for _ in range(1000):
        rng.shuffle(other_nodes)
        order = other_nodes + [target_node]
        candidates = [
            (order[i], order[j])
            for i in range(len(order))
            for j in range(i + 1, len(order))
        ]
        novel = [edge for edge in candidates if edge not in original_edges]
        if len(novel) >= edge_count:
            chosen = rng.sample(novel, edge_count)
            target_parents = [edge for edge in novel if edge[1] == target_node]
            if not any(edge[1] == target_node for edge in chosen):
                if target_parents:
                    replacement = rng.choice(target_parents)
                else:
                    # If every possible target parent is already in the input,
                    # retain one such edge and randomize the remaining topology.
                    replacement = rng.choice([edge for edge in original_edges if edge[1] == target_node])
                chosen[-1] = replacement
        else:
            chosen = rng.sample(candidates, edge_count)
            if not any(edge[1] == target_node for edge in chosen):
                target_candidates = [edge for edge in candidates if edge[1] == target_node]
                replacement = rng.choice(target_candidates)
                chosen[-1] = replacement
        chosen_set = set(chosen)
        if len(chosen_set) == edge_count and chosen_set != original_edges:
            break
    else:
        raise ValueError("Could not create a distinct DAG with the same edge count and target as a sink")

    perturbed = pd.DataFrame(0, index=nodes, columns=nodes, dtype=int)
    for source, destination in chosen_set:
        perturbed.loc[source, destination] = 1

    output_dag = nx.DiGraph()
    output_dag.add_nodes_from(nodes)
    output_dag.add_edges_from(chosen_set)
    if not nx.is_directed_acyclic_graph(output_dag):
        raise RuntimeError("Internal error: generated topology is not acyclic")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        pickle.dump(perturbed, stream)

    shared = len(original_edges & chosen_set)
    metadata = {
        "input_graph": str(input_path),
        "output_graph": str(output_path),
        "seed": seed,
        "target_node": target_node,
        "node_count": len(nodes),
        "edge_count_original": edge_count,
        "edge_count_perturbed": len(chosen_set),
        "shared_directed_edges": shared,
        "removed_original_edges": edge_count - shared,
        "added_new_edges": len(chosen_set) - shared,
        "edge_overlap_fraction": shared / edge_count,
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset label for run documentation")
    parser.add_argument("--input", required=True, type=Path, help="Pickled pandas adjacency matrix")
    parser.add_argument("--output", required=True, type=Path, help="Destination pickle path")
    parser.add_argument("--seed", required=True, type=int, help="Random topology seed")
    parser.add_argument("--target-node", default=None, help="Sink node; defaults to the final graph label")
    args = parser.parse_args()
    print(f"Dataset: {args.dataset}")
    perturb_graph(args.input, args.output, args.seed, args.target_node)


if __name__ == "__main__":
    main()
