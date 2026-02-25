from dmst.ga import GAConfig, run_ga_dmst


def main():
    print("Hello from dmst!")
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

    config = GAConfig(
        population_size=50, generations=200, mutation_rate=0.1, tournament_size=4
    )

    print("=" * 31)
    print("Running dmst with:")
    print(f"    n_nodes:       {n_nodes}")
    print(f"    max_d:         {max_d}")
    print(f"    weights:       {weights}")
    print(f"    reliabilities: {weights}")
    print("GA config:")
    print(f"    {config}")
    print("=" * 31, "\n")

    pareto_front, fitness_front = run_ga_dmst(
        n_nodes, weights, reliabilities, max_d, config=config
    )

    # print("\n", "=" * 11, " Results ", "=" * 11, sep="")
    # print("Pareto Front:")
    # print(f"    {pareto_front}")
    # print(f"    {fitness_front}")
    # print("=" * 31)


if __name__ == "__main__":
    main()
