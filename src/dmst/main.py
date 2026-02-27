import pprint

from dmst.ga import DMSTGraph, GAConfig, run_ga_dmst


def main():
    weights = {
        (0, 3): 10,
        (1, 3): 15,
        (2, 3): 20,
        (3, 4): 5,
        (4, 5): 30,
        (0, 1): 50,
        (1, 2): 40,
        (2, 4): 25,
        (0, 5): 60,
    }

    reliabilities = {
        (0, 3): 0.1,
        (1, 3): 0.76,
        (2, 3): 0.98,
        (3, 4): 0.5,
        (4, 5): 0.2,
        (0, 1): 0.4,
        (1, 2): 0.69,
        (2, 4): 0.77,
        (0, 5): 0.33,
    }

    n_nodes = 6
    max_d = 3

    graph = DMSTGraph(weights, reliabilities)

    config = GAConfig(
        population_size=50, generations=100, mutation_rate=0.1, tournament_size=2
    )

    print("=" * 25)
    print("Running dmst with:")
    print(f"- n_nodes: {n_nodes}")
    print(f"- max_d:   {max_d}")
    print(f"- weights:       \n{pprint.pformat(weights, indent=2)}")
    print(f"- reliabilities: \n{pprint.pformat(reliabilities, indent=2)}")
    print("- GA config:")
    print(f"{pprint.pformat(config.__dict__, indent=2)}")
    print("=" * 25, "\n")

    pareto_front, fitness_front, _, hist = run_ga_dmst(
        n_nodes, graph, max_d, config=config
    )


if __name__ == "__main__":
    main()
