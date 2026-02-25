from dataclasses import dataclass
import math
import random
from dmst.prufer import Prufer, edges_from_tree, prufer_decode

type Weights = dict[tuple[int, int], int]
type Reliabilities = dict[tuple[int, int], float]
type Population = list[Prufer]
type PopulationFitness = list[tuple[float, float]]


@dataclass
class GAConfig:
    population_size: int
    generations: int
    mutation_rate: float = 0.05
    tournament_size: int = 3
    penalty: int = 10_000


__DEFAULT_GACONFIG: GAConfig = GAConfig(
    population_size=100, generations=200, mutation_rate=0.05, tournament_size=3
)


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
    weights: Weights,
    reliabilities: Reliabilities,
    max_d: int,
    config: GAConfig = __DEFAULT_GACONFIG,
) -> tuple[Population, PopulationFitness]:
    population = generate_random_population(config.population_size, n_nodes)

    for gen in range(config.generations):
        fitness_population = [
            fitness(prufer, weights, reliabilities, max_d) for prufer in population
        ]
        pareto_front, fitness_front = get_pareto_front(population, fitness_population)

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

    return final_front, final_fitness_front


if __name__ == "__main__":
    pop_example = ["A", "B", "C", "D"]
    fit_example = [
        (1000, 5.0),  # A: Muy barato, alto riesgo
        (1200, 4.0),  # B: Buen equilibrio
        (1500, 4.5),  # C: Dominado por B (B es más barato y tiene menos riesgo)
        (2000, 2.0),  # D: Muy caro, mínimo riesgo
        (1100, 6.0),  # E: Dominado por A (A es más barato y tiene menos riesgo)
    ]

    best_inds, best_fitness = get_pareto_front(pop_example, fit_example)

    for ex, fit in zip(best_inds, best_fitness):
        cost, risk = fit
        print(f"- {ex} with Cost {cost} and Risk {risk}")
