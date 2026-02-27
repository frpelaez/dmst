import math
import random
from dataclasses import dataclass

from dmst.prufer import Prufer, edges_from_tree, prufer_decode


type Weights = dict[tuple[int, int], int]
type Reliabilities = dict[tuple[int, int], float]
type Population = list[Prufer]
type PopulationFitness = list[tuple[float, float]]


@dataclass
class DMSTGraph:
    """
    Object that represents agraph in the context of the d-MSTr problem
    """

    weights: Weights
    reliabilities: Reliabilities

    def n_nodes(self) -> int:
        return 1 + max(edge[0] for edge in self.weights)


@dataclass
class GAConfig:
    """
    Object that encapsulates the configuration values for the Genetic Algorithm
    """

    population_size: int
    generations: int
    mutation_rate: float = 0.05
    tournament_size: int = 3
    penalty: int = 10_000

    @staticmethod
    def default() -> "GAConfig":
        return GAConfig(
            population_size=100, generations=200, mutation_rate=0.05, tournament_size=3
        )


@dataclass
class HistoryResults:
    """
    Object that centralizes the evolution of the metrics (avg & min weights and risks) of a Genetic Algorithm execution
    """

    mean_weights: list[float]
    mean_risks: list[float]
    best_weights: list[float]
    best_risks: list[float]


def fitness(
    prufer: Prufer,
    weights: Weights,
    reliabilities: Reliabilities,
    max_d: int,
    penalty: int = 10_000,
) -> tuple[float, float]:
    edges = edges_from_tree(prufer_decode(prufer))

    total_weight = 0
    total_log_risk = 0.0

    for u, v in edges:
        edge = (u, v) if (u, v) in weights else (v, u)
        if edge in weights:
            total_weight += weights[edge]
            p = reliabilities[edge]
            p = max(p, 1e-9)
            total_log_risk -= math.log(p)
        else:
            total_weight += penalty
            total_log_risk += penalty

    excesses = 0
    n_nodes = len(prufer) + 2

    for node in range(n_nodes):
        deg = prufer.count(node) + 1
        if deg > max_d:
            excesses += deg - max_d

    final_weight = total_weight + penalty * excesses
    final_log_risk = total_log_risk + penalty * excesses

    return (final_weight, final_log_risk)


def pareto_compare(
    fitness_a: tuple[float, float], fitness_b: tuple[float, float]
) -> bool:
    """
    Compares two fitness tuples (weight_a, risk_a) and (weight_b, risk_b). Returns `True`
    if and only if `a` dominates `b`, ie, `a` is at least as good as `b` (doest not have higher
    weight and doest not have higher risk) and `a` is strictly better than `b` in at least
    one field
    """
    weight_a, risk_a = fitness_a
    weight_b, risk_b = fitness_b

    a_not_worse = weight_a <= weight_b and risk_a <= risk_b
    a_strict_better = weight_a < weight_b or risk_a < risk_b

    return a_not_worse and a_strict_better


def get_pareto_front(
    population: list[Prufer], fitness_population: list[tuple[float, float]]
) -> tuple[Population, PopulationFitness]:
    pareto_front = []
    fitness_front = []
    for i in range(len(population)):
        is_dominated = False
        for j in range(len(population)):
            if i == j:
                continue
            if pareto_compare(fitness_population[j], fitness_population[i]):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append(population[i])
            fitness_front.append(fitness_population[i])

    return pareto_front, fitness_front


def generate_random_population(population_size: int, n_nodes: int) -> Population:
    prufer_length = n_nodes - 2
    population = []
    for _ in range(population_size):
        prufer = [random.randint(0, n_nodes - 1) for _ in range(prufer_length)]
        population.append(prufer)

    return population


def tournament_selection(
    population: Population, fitness_population: list[float], k: int = 3
) -> Prufer:
    selected = random.sample(list(zip(population, fitness_population)), k)
    winner = min(selected, key=lambda x: x[1])[0]
    return winner


def tournament_selection_pareto(
    population: Population,
    fitness_population: PopulationFitness,
    k: int = 3,
) -> Prufer:
    selected = random.sample(list(range(len(population))), k)
    candidates = []
    for i in selected:
        is_dominated = False
        for j in selected:
            if i == j:
                continue
            if pareto_compare(fitness_population[j], fitness_population[i]):
                is_dominated = True
                break
        if not is_dominated:
            candidates.append(population[i])

    return random.choice(candidates)


def crossover(parent1: Prufer, parent2: Prufer) -> tuple[Prufer, Prufer]:
    division_point = random.randint(0, len(parent1) - 1)
    offspring1 = parent1[:division_point] + parent2[division_point:]
    offspring2 = parent2[:division_point] + parent1[division_point:]
    return offspring1, offspring2


def mutation(prufer: Prufer, mutation_probability: float = 0.05) -> Prufer:
    mutated = prufer.copy()
    for i in range(len(prufer)):
        if random.random() < mutation_probability:
            mutated[i] = random.randint(0, len(prufer) + 1)

    return mutated


def run_ga_dmst(
    n_nodes: int,
    graph: DMSTGraph,
    max_d: int,
    config: GAConfig = GAConfig.default(),
) -> tuple[Population, PopulationFitness, PopulationFitness, HistoryResults]:
    weights = graph.weights
    reliabilities = graph.reliabilities

    history_mean_weights = []
    history_mean_risks = []
    history_best_weights = []
    history_best_risks = []

    population = generate_random_population(config.population_size, n_nodes)

    for gen in range(config.generations):
        fitness_population = [
            fitness(prufer, weights, reliabilities, max_d) for prufer in population
        ]
        pareto_front, fitness_front = get_pareto_front(population, fitness_population)

        valid_fits = [f for f in fitness_front if f[0] < config.penalty]
        if valid_fits:
            mean_weight = sum(f[0] for f in valid_fits) / len(valid_fits)
            mean_risk = sum(f[1] for f in valid_fits) / len(valid_fits)
            min_weight = min(f[0] for f in valid_fits)
            min_risk = min(f[1] for f in valid_fits)
        else:
            mean_weight = (
                history_mean_weights[-1] if history_mean_weights else config.penalty
            )
            mean_risk = history_mean_risks[-1] if history_mean_risks else config.penalty
            min_weight = (
                history_best_weights[-1] if history_best_weights else config.penalty
            )
            min_risk = history_best_risks[-1] if history_best_risks else config.penalty

        history_mean_weights.append(mean_weight)
        history_mean_risks.append(mean_risk)
        history_best_weights.append(min_weight)
        history_best_risks.append(min_risk)

        if gen % 20 == 0 or gen == config.generations - 1:
            print(
                f"Gen {gen + 1:>3} | Solutions in the current Pareto Front: {len(pareto_front)}"
            )

        new_population = []
        new_population.extend(pareto_front[: config.population_size])
        while len(new_population) < config.population_size:
            parent1 = tournament_selection_pareto(
                population, fitness_population, k=config.tournament_size
            )
            parent2 = tournament_selection_pareto(
                population, fitness_population, k=config.tournament_size
            )

            offspring1, offspring2 = crossover(parent1, parent2)

            offspring1 = mutation(offspring1, mutation_probability=config.mutation_rate)
            offspring2 = mutation(offspring2, mutation_probability=config.mutation_rate)

            new_population.extend([offspring1, offspring2])

        population = new_population[: config.population_size]

    final_fitness = [
        fitness(prufer, weights, reliabilities, max_d, config.penalty)
        for prufer in population
    ]
    final_front, final_fitness_front = get_pareto_front(population, final_fitness)

    population_valid_fitness = [f for f in final_fitness if f[0] < config.penalty]

    hist_results = HistoryResults(
        history_mean_weights,
        history_mean_risks,
        history_best_weights,
        history_best_risks,
    )

    return final_front, final_fitness_front, population_valid_fitness, hist_results


if __name__ == "__main__":
    pop_example = ["A", "B", "C", "D"]
    fit_example = [
        (1000.0, 5.0),
        (1200.0, 4.0),
        (1500.0, 4.5),
        (2000.0, 2.0),
        (1100.0, 6.0),
    ]

    best_inds, best_fitness = get_pareto_front(pop_example, fit_example)  # type: ignore

    for ex, fit in zip(best_inds, best_fitness):
        cost, risk = fit
        print(f"- {ex} with Cost {cost} and Risk {risk}")
