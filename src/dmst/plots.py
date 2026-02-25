import matplotlib.pyplot as plt


def plot_pareto_front(
    fitness_population: list[tuple[float, float]],
    fitness_front: list[tuple[float, float]],
):
    weights_pop = [fit[0] for fit in fitness_population]
    risks_pop = [fit[1] for fit in fitness_population]

    sorted_front = sorted(fitness_front, key=lambda x: x[0])

    weights_frt = [fit[0] for fit in sorted_front]
    risks_frt = [fit[1] for fit in sorted_front]

    plt.figure(figsize=(10, 6))

    plt.scatter(
        weights_pop,
        risks_pop,
        color="lightgray",
        label="Final population (Dominated)",
        alpha=0.6,
    )

    plt.scatter(
        weights_frt,
        risks_frt,
        color="red",
        label="Pareto's Front",
        s=50,
        zorder=5,
    )

    plt.plot(weights_frt, risks_frt, color="red", linestyle="--", alpha=0.7, zorder=4)

    plt.title("Multiobjective d-MST (Weights vs Risk)", fontsize=14)
    plt.xlabel("Total weight of the network (minimize)", fontsize=12)
    plt.ylabel("Failure risk of the network [-ln(p_fail)] (minimize)", fontsize=12)

    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    import random
    from dmst.ga import get_pareto_front

    a = [(i, i + 1) for i in range(100)]
    pop = [(random.gauss(3000, 500), random.gauss(5, 2)) for _ in range(100)]

    _, front = get_pareto_front(a, pop)

    plot_pareto_front(pop, front)
