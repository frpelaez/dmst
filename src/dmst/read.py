import csv

from dmst.ga import DMSTGraph


def load_graph_from_csv(filepath: str) -> tuple[DMSTGraph, int]:
    weights, reliabilities = {}, {}
    visited = set()

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for line in reader:
                u = int(line["u"])
                v = int(line["v"])
                w = float(line["weight"])
                p = float(line["prob"])
                weights[(u, v)] = w
                reliabilities[(u, v)] = p
                visited.add(u)
                visited.add(v)

        n_nodes = max(visited)
        print(
            f"[INFO] Graph successfully loaded from CSV: {len(weights)} edges and {n_nodes} nodes"
        )
        return DMSTGraph(weights, reliabilities), n_nodes
    except Exception as e:
        print(f"[ERROR] Unnable to read CSV: {e}")
        exit(1)
