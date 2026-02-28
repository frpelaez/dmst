from dmst.ga import HistoryResults
import os
import matplotlib.pyplot as plt


def plot_pareto_front(
    fitness_population: list[tuple[float, float]],
    fitness_front: list[tuple[float, float]],
    output_dir: str | None = None,
    show_plot: bool = True,
):
    """
    Scatter plot to visualize the Pareto front (set of non dominated solutions)
    in the objective function 2D space.
    """
    weights_pop = [fit[0] for fit in fitness_population]
    risks_pop = [fit[1] for fit in fitness_population]

    sorted_front = sorted(fitness_front, key=lambda x: x[0])

    weights_frt = [fit[0] for fit in sorted_front]
    risks_frt = [fit[1] for fit in sorted_front]

    plt.figure(figsize=(10, 6))

    plt.scatter(
        weights_pop,
        risks_pop,
        color="gray",
        label="Final population (Dominated)",
        alpha=0.6,
    )

    plt.scatter(
        weights_frt,
        risks_frt,
        color="teal",
        label="Pareto Front",
        s=50,
        zorder=5,
    )

    plt.plot(weights_frt, risks_frt, color="teal", linestyle="--", alpha=0.7, zorder=4)

    plt.title("Multiobjective d-MST (Weights vs Risk)", fontsize=14)
    plt.xlabel("Total weight of the network", fontsize=12)
    plt.ylabel("Failure risk of the network [-ln(p_fail)]", fontsize=12)

    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        count = len(os.listdir(output_dir)) + 1
        filepath = os.path.join(output_dir, f"pareto_front_result_{count}.png")
        plt.savefig(filepath, dpi=300)
        print(f"Plot successfully saved to {filepath}")
        if show_plot:
            plt.show()
    else:
        plt.show()

    plt.close()


def plot_history(
    history: HistoryResults,
    output_dir: str | None = None,
    show_plot: bool = True,
):
    """
    Line plots to see the evolution of the average and minimum weight and risk in the population
    """
    history_mean_weight = history.mean_weights
    history_mean_risk = history.mean_risks
    history_best_weight = history.best_weights
    history_best_risk = history.best_risks

    generations = list(range(len(history.mean_weights)))
    start = len(generations) - len(history_mean_weight)
    generations = generations[start:]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Generation")

    weight_color = "tab:blue"
    ax1.set_ylabel("Weight", color=weight_color, weight="bold")
    (weight_line_mean,) = ax1.plot(
        generations,
        history_mean_weight,
        color=weight_color,
        linewidth=2,
        label="Mean weight",
    )
    (weight_line_best,) = ax1.plot(
        generations,
        history_best_weight,
        color=weight_color,
        linewidth=1.5,
        linestyle="--",
        label="Best weight",
    )
    ax1.tick_params(axis="y", labelcolor=weight_color)

    ax2 = ax1.twinx()
    risk_color = "tab:red"
    ax2.set_ylabel("Risk", color=risk_color, weight="bold")
    (risk_line_mean,) = ax2.plot(
        generations,
        history_mean_risk,
        color=risk_color,
        linewidth=2,
        label="Mean risk",
    )
    (risk_line_best,) = ax2.plot(
        generations,
        history_best_risk,
        color=risk_color,
        linewidth=1.5,
        linestyle="--",
        label="Best risk",
    )
    ax2.tick_params(axis="y", labelcolor=risk_color)

    ax1.grid(True, linestyle=":", alpha=0.6)

    plt.title("Convergence of Genetic Algorithm for d-MSTr", fontsize=14)
    fig.legend(
        handles=[weight_line_mean, risk_line_mean, weight_line_best, risk_line_best],
        loc="upper right",
        bbox_to_anchor=(0.9, 0.92),
        ncol=2,
    )

    fig.tight_layout()

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        count = len(os.listdir(output_dir)) + 1
        path = os.path.join(output_dir, f"convergence_{count}.png")
        plt.savefig(path, dpi=300)
        print(f"Convergence plot successfully saved to {path}")
        if show_plot:
            plt.show()
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    import random
    from dmst.ga import get_pareto_front

    a = [(i, i + 1) for i in range(100)]
    pop = [(random.gauss(3000, 500), random.gauss(5, 2)) for _ in range(100)]

    _, front = get_pareto_front(a, pop)  # type: ignore

    plot_pareto_front(pop, front)
