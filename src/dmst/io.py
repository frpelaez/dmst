import csv
from typing import Any

from dmst.ga import DMSTGraph, GAConfig


def load_graph_from_csv(filepath: str) -> tuple[DMSTGraph, int]:
    """
    Load a dMSTr graph (`n_nodes` and `weights`, `reliabilities` matrices) from a CSV file
    """
    weights, reliabilities = {}, {}
    visited = set()

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for line in reader:
                u = int(line["from"])
                v = int(line["to"])
                w = float(line["weight"])
                p = float(line["prob"])
                weights[(u, v)] = w
                reliabilities[(u, v)] = p
                visited.add(u)
                visited.add(v)

        n_nodes = max(visited) + 1
        print(
            f"[IO-INFO] Graph successfully loaded from CSV: {len(weights)} edges and {n_nodes} nodes"
        )
        return DMSTGraph(weights, reliabilities), n_nodes
    except Exception as e:
        print(f"[IO-ERROR] Unnable to read CSV: {e}")
        exit(1)


def save_graph_to_csv(filepath: str, n_nodes: int, graph: DMSTGraph):
    """
    Save a dMSTr graph to a file for experiment reproductibility
    """
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["from", "to", "weight", "prob"])
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                writer.writerow(
                    [i, j, graph.weights[(i, j)], graph.reliabilities[(i, j)]]
                )
    print(f"[IO-INFO] Graph successfully saved to: {filepath}")


def load_config_from_json(filepath: str) -> tuple[GAConfig, dict[str, Any]]: ...
