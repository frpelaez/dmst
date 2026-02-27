import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(path, "src"))


import argparse
import random
from argparse import ArgumentParser

from dmst.ga import DMSTGraph, GAConfig, run_ga_dmst
from dmst.io import load_graph_from_csv, save_graph_to_csv
from dmst.plots import plot_history, plot_pareto_front


def generate_random_graph(n_nodes: int) -> DMSTGraph:
    weights, reliabilities = {}, {}
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            weights[(u, v)] = random.randint(10, 100)
            reliabilities[(u, v)] = random.uniform(0.65, 0.95)

    return DMSTGraph(weights, reliabilities)


def get_parser() -> ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Multiobjective Genetic Algorithm for d-MSTr problem"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to the output directory for the generated plot(s). If omited, the plot(s) will be shown instead",
    )

    parser.add_argument(
        "-n",
        "--n_nodes",
        type=int,
        default=50,
        help="Number of nodes",
    )

    parser.add_argument(
        "-d",
        "--d_max",
        type=int,
        default=3,
        help="Degree restriction for the nodes of the MST solutions",
    )

    parser.add_argument(
        "-g",
        "--generations",
        type=int,
        default=None,
        help="Number of generations",
    )

    parser.add_argument(
        "-s",
        "--pop_size",
        type=int,
        default=None,
        help="Population size",
    )

    parser.add_argument(
        "-m",
        "--mutation_rate",
        type=int,
        default=None,
        help="Mutation rate",
    )

    parser.add_argument(
        "-t",
        "--tournament_size",
        type=int,
        default=None,
        help="Tournament size",
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Input CSV file containing [u, v, weight, prob] entries. If omited, a random complete graph is used instead",
    )

    parser.add_argument(
        "-p",
        "--plots",
        action="store_true",
        default=False,
        help="Whether to show the generated plot(s)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Whether to show the other arguments captured by the parser",
    )

    parser.add_argument(
        "--save_graph",
        type=str,
        default=None,
        help="Save the graph used for the demonstration to the provided file",
    )

    return parser


def main():
    args = get_parser().parse_args()

    random.seed(42)

    n_nodes = args.n_nodes
    d_max = args.d_max
    config = GAConfig(
        population_size=100, generations=200, mutation_rate=0.05, tournament_size=3
    )

    if args.input:
        graph, n_nodes = load_graph_from_csv(args.input)
    else:
        graph = generate_random_graph(n_nodes)

    if args.generations:
        config.generations = args.generations

    if args.pop_size:
        config.population_size = args.pop_size

    if args.mutation_rate:
        config.mutation_rate = args.mutation_rate

    if args.tournament_size:
        config.tournament_size = args.tournament_size

    if args.save_graph:
        save_graph_to_csv(args.save_graph, n_nodes, graph)

    print("Starting Multiobjective Genetic Algorithm for d-MSTr")

    if args.verbose:
        print("=" * 25)
        print("dMST args:")
        print(f"    n_nodes:      {n_nodes}")
        print(f"    d_max:        {d_max}")
        print(
            f"    graph:        {'random-generated' if not args.input else args.input}"
        )
        print("GA config:")
        print(f"    pop_size:     {config.population_size}")
        print(f"    generations:  {config.generations}")
        print(f"    mut_rate:     {config.mutation_rate}")
        print(f"    tournament_k: {config.tournament_size}")
        print("Extra args:")
        print(f"    output_dir:   {args.output}")
        print(f"    show_plots:   {not args.output or args.plots}")
        print(f"    save_graph:   {args.save_graph}")
        print("=" * 25)

    front, fitness_front, fitness_pop, history = run_ga_dmst(
        n_nodes, graph, d_max, config
    )

    print(f"Evolution finished: {len(front)} solutions found in the Pareto Front")

    # for i, fit in enumerate(fitness_front):
    #     print(f"Solution {i + 1:>2}: Weight = {fit[0]} | Risk = {fit[1]:4f}")

    print("Generating plots...")
    plot_pareto_front(
        fitness_pop, fitness_front, output_dir=args.output, show_plot=args.plots
    )
    plot_history(
        history,
        output_dir=args.output,
        show_plot=args.plots,
    )


if __name__ == "__main__":
    main()
