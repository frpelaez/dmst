"""
Definitions for labeled trees and Prüfer encoding

- Types:
    `Tree`:   dict<int, set<int>>
    `Prufer`: list<int>

- Functions:
    `tree_from_edges`: (list<tuple <int, int>>, int) -> `Tree`
    `edges_from_tree`: (`Tree`) -> list<tuple <int, int>>
    `prufer_encode`:   (`Tree`, int) -> `Prufer`
    `tree_from_edges`: (`Prufer`) -> `Tree`
"""

type Tree = dict[int, set[int]]
type Prufer = list[int]


def tree_from_edges(edges: list[tuple[int, int]], n_nodes: int) -> Tree:
    """
    `tree_from_edges` builds a `Tree` (represented by adjacency lists as `dict[int, set[int]]`)
    from a list of edges, each one of the edges is represented as a pair (int, int)
    """
    tree = {i: set() for i in range(n_nodes)}
    for u, v in edges:
        tree[u].add(v)
        tree[v].add(u)

    return tree


def edges_from_tree(tree: Tree) -> list[tuple[int, int]]:
    """
    `edges_from_tree` builds a list of edges representing a `Tree`
    """
    edges = []
    for node, neighbors in tree.items():
        for n in neighbors:
            if node < n:
                edges.append((node, n))

    return edges


def prufer_encode(tree: Tree, n_nodes: int) -> Prufer:
    """
    `prufer_encode` takes a `Tree` and encodes it as a `Prüfer Sequence`
    """
    prufer = []
    for _ in range(n_nodes - 2):
        leaves = [node for node, neighbors in tree.items() if len(neighbors) == 1]
        min_leaf = min(leaves)
        neighbor = list(tree[min_leaf])[0]
        prufer.append(neighbor)
        tree[neighbor].remove(min_leaf)
        del tree[min_leaf]

    return prufer


def prufer_decode(prufer: Prufer) -> Tree:
    """
    `prufer_decode` takes a `Prüfer Sequence` and decodes it to produce its corresponding `Tree`
    """
    n_nodes = len(prufer) + 2
    available_nodes = list(range(n_nodes))
    edges = []
    current_prufer = prufer.copy()
    while current_prufer:
        leaf = min(node for node in available_nodes if node not in current_prufer)
        neighbor = current_prufer.pop(0)
        edges.append((leaf, neighbor))
        available_nodes.remove(leaf)

    edges.append((available_nodes[0], available_nodes[1]))

    return tree_from_edges(edges, n_nodes)


if __name__ == "__main__":
    n = 6
    edges = [(0, 3), (1, 3), (2, 3), (3, 4), (4, 5)]
    tree = tree_from_edges(edges, n)

    prufer = prufer_encode(tree, n)

    print(f"Tree:          {edges}")
    print(f"Prufer encode: {prufer}")

    decoded_tree = prufer_decode(prufer)

    print(f"Decoded tree:  {edges_from_tree(decoded_tree)}")
