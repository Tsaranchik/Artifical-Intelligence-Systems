import numpy as np
import random
import os
import math
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Callable


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N = 8 # field size
# crop name, yield per hectar in tons, cost per ton in dollars
CROPS = [
	("Wheat", 3.21, 173.19),
	("Corn", 6.05, 195.72),
	("Soy", 3.16, 383.52),
	("Barley", 3.10, 117.83),
	("Grapes", 22.5, 600),
	("Rice", 4.5, 450),
	("Sugar beet", 30, 550), # nerfed (yield 50 -> 30)
	("Potato", 21, 230),
	("Cassava", 10, 200)
]
k = len(CROPS)

POP_SIZE = 20
GENERATIONS = 50
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 3

RESULTS_DIR = "ga_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_fields(
		N: int,
		crops: list[tuple[str, int, int]],
		seed: int = SEED
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]: 
	random.seed(seed)
	np.random.seed(seed)
	crop_names = [c[0] for c in crops]
	base_yields = np.array([c[1] for c in crops])
	crop_costs = np.array([c[2] for c in crops])

	areas = np.round(np.random.uniform(1.0, 10.0, size=N), 2)
	soil_factors = np.round(np.random.uniform(0.7, 1.3, size=N), 3)

	yields_matrix = np.zeros((N, len(crops)))
	costs_matrix = np.zeros((N, len(crops)))
	for i in range(N):
		for j in range(len(crops)):
			yields_matrix[i, j] = areas[i] * base_yields[j] * soil_factors[i]
			costs_matrix[i, j] = areas[i] * crop_costs[j]
	fields_df = pd.DataFrame({
		"field_id": np.arange(N),
		"area": areas,
		"soil_factor": soil_factors
	})

	return fields_df, yields_matrix, costs_matrix, crop_names


def total_yield_and_cost(
		solution: np.ndarray,
		yields_matrix: np.ndarray,
		costs_matrix: np.ndarray
) -> tuple[float, float]:
	n = len(solution)
	idx = (np.arange(n), solution)
	total_yield = float(yields_matrix[idx].sum())
	total_cost = float(costs_matrix[idx].sum())

	return total_yield, total_cost


def compute_theoretical_bounds(
		yields_matrix: np.ndarray,
		costs_matrix: np.ndarray
) -> tuple[float, float, float, float]:
	min_yield = float(yields_matrix.min(axis=1).sum())
	max_yield = float(yields_matrix.max(axis=1).sum())
	min_cost = float(costs_matrix.min(axis=1).sum())
	max_cost = float(costs_matrix.max(axis=1).sum())

	return (min_yield, max_yield, min_cost, max_cost)


def fitness_of_solution(
		solution: np.ndarray,
		yields_matrix: np.ndarray,
		costs_matrix: np.ndarray,
		mode: str = "weighted",
		w: float = 0.5,
		bounds: Optional[tuple[float, float, float, float]] = None
) -> float:
	total_y, total_c = total_yield_and_cost(solution, yields_matrix, costs_matrix)
	if bounds is None:
		bounds = compute_theoretical_bounds(yields_matrix, costs_matrix)
	min_y, max_y, min_c, max_c = bounds
	ny = (total_y - min_y) / (max_y - min_y) if (max_y - min_y) > 0 else 0.5
	nc = (total_c - min_c) / (max_c - min_c) if (max_c - min_c) > 0 else 0.5
	
	return ny * math.exp(-w * nc)


def one_point_crossover(
		parent1: np.ndarray,
		parent2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
	n = len(parent1)
	if n < 2:
		return parent1.copy(), parent2.copy()
	cp = random.randint(1, n-1)
	child1 = np.concatenate([parent1[:cp], parent2[cp:]])
	child2 = np.concatenate([parent2[:cp], parent1[cp:]])

	return child1, child2


def two_point_crossover(
		parent1: np.ndarray,
		parent2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
	n = len(parent1)
	if n < 3:
		return one_point_crossover(parent1, parent2)
	a = random.randint(1, n-2)
	b = random.randint(a+1, n-1)
	child1 = parent1.copy()
	child2 = parent2.copy()
	child1[a:b], child2[a:b] = parent2[a:b], parent1[a:b]

	return child1, child2


def uniform_crossover(
		parent1: np.ndarray,
		parent2: np.ndarray,
		p: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
	n = len(parent1)
	mask = np.random.rand(n) < p
	child1 = np.where(mask, parent1, parent2)
	child2 = np.where(mask, parent2, parent1)
	
	return child1, child2


def random_reset_mutation(
		child: list[int],
		k: int,
		mutation_rate: float
) -> list[int]:
	for i in range(len(child)):
		if random.random() < mutation_rate:
			new = random.randrange(k)
			while new == child[i] and k > 1:
				new = random.randrange(k)
			child[i] = new
	
	return child


def swap_mutation(
		child: list[int],
		k: int,
		mutation_rate: float
) -> list[int]:
	if random.random() < mutation_rate and len(child) >= 2:
		i, j = random.sample(range(len(child)), 2)
		child[i], child[j] = child[j], child[i]
	
	return child


def scramble_mutation(
		child: list[int],
		k: int,
		mutation_rate: float
) -> list[int]:
	n = len(child)
	if random.random() < mutation_rate and n >= 3:
		a = random.randint(0, n-2)
		b = random.randint(a+1, n-1)
		sub = child[a:b+1].copy()
		np.random.shuffle(sub)
		child[a:b+1] = sub
	
	return child


def tournament_selection(
		pop: list[np.ndarray],
		fintesses: list[float],
		tournament_size: int
) -> np.ndarray:
	i_idxs = np.random.randint(0, len(pop), size=tournament_size)
	best_idx = i_idxs[0]
	best_fit = fintesses[best_idx]
	for idx in i_idxs[1:]:
		if fintesses[idx] > best_fit:
			best_fit = fintesses[idx]
			best_idx = idx
	
	return pop[best_idx].copy()


class Individual:
	def __init__(
			self,
			genome: np.ndarray,
			yields_matrix: np.ndarray,
			costs_matrix: np.ndarray
	) -> None:
		self.genome = genome
		self.yield_val, self.cost_val = total_yield_and_cost(
			genome,
			yields_matrix,
			costs_matrix
		)
		self.fitness = fitness_of_solution(genome, yields_matrix, costs_matrix)
	
	def __repr__(self):
		return f"Individual(Y={self.yield_val:.2f}, C={self.cost_val:.2f}, F={self.fitness:.4f})"


def create_random_individual(
		yields_matrix: np.ndarray,
		costs_matrix: np.ndarray
) -> Individual:
	genome = np.random.randint(0, k, size=N)
	return Individual(genome, yields_matrix, costs_matrix)


def create_initial_pop(
		pop_size: int,
		yields_matrix: np.ndarray,
		costs_matrix: np.ndarray
) -> list[Individual]:
	return [create_random_individual(yields_matrix, costs_matrix) for _ in range(pop_size)]


def evolve_pop(
		pop: list[Individual],
		yields_matrix: np.ndarray,
		costs_matrix: np.ndarray,
		crossover_fn: Callable,
		mutation_fn: Callable,
		crossover_rate: float = CROSSOVER_RATE,
		mutation_rate: float = MUTATION_RATE
) -> list[Individual]:
	fitnesses = [ind.fitness for ind in pop]
	genomes = [ind.genome for ind in pop]
	new_pop = []
	
	best_individual = pop[np.argmax(fitnesses)]
	new_pop.append(best_individual)

	while len(new_pop) < POP_SIZE:
		parent1 = tournament_selection(genomes, fitnesses, TOURNAMENT_SIZE)
		parent2 = tournament_selection(genomes, fitnesses, TOURNAMENT_SIZE)
		if random.random() < crossover_rate:
			child1, child2 = crossover_fn(parent1, parent2)
		else:
			child1, child2 = parent1.copy(), parent2.copy()
		
		child1 = mutation_fn(child1.tolist(), k, mutation_rate)
		child2 = mutation_fn(child2.tolist(), k, mutation_rate)

		new_pop.append(Individual(np.array(child1), yields_matrix, costs_matrix))
		if len(new_pop) < POP_SIZE:
			new_pop.append(Individual(np.array(child2), yields_matrix, costs_matrix))

	return new_pop[:POP_SIZE]


def run_experiment(
		crossover_name: str,
		crossover_fn: Callable,
		mutation_name: str,
		mutation_fn: Callable,
		yileds_matrix: np.ndarray,
		costs_matrix: np.ndarray,
		generations: int = GENERATIONS
) -> tuple[list[float], Individual]:
	pop = create_initial_pop(POP_SIZE, yileds_matrix, costs_matrix)
	best_fitness_history = []

	for gen in range(generations):
		pop = evolve_pop(pop, yileds_matrix, costs_matrix, crossover_fn, mutation_fn)
		best_fitness = max(ind.fitness for ind in pop)
		best_fitness_history.append(best_fitness)
	
	best_individual = max(pop, key=lambda ind: ind.fitness)
	return best_fitness_history, best_individual


def main_experiment():
	fields_df, yields_matrix, costs_matrix, crop_names = generate_fields(N, CROPS)

	print(
		f"Fields param:\n {fields_df}\n"
		f"\nYields matrix (fields x crops):\n {yields_matrix}\n"
		f"\nCosts matrix (fields x crops):\n {costs_matrix}"
	)

	crossovers = {
		"One-point": one_point_crossover,
		"Two-point": two_point_crossover,
		"Uniform": uniform_crossover
	}
	mutations = {
		"Random Reset": random_reset_mutation,
		"Swap": swap_mutation,
		"Scramble": scramble_mutation
	}

	plt.figure(figsize=(12, 8))
	results = {}
	
	for crossover_name, crossover_fn in crossovers.items():
		for mutation_name, mutation_fn in mutations.items():
			combo_name = f"{crossover_name} + {mutation_name}"
			print(f"\n--- Starting experiment: {combo_name} ---")

			fitnesses_history, best_individual = run_experiment(
				crossover_name, crossover_fn, mutation_name, mutation_fn,
				yields_matrix, costs_matrix
			)

			results[combo_name] = {
				"fitness_history": fitnesses_history,
				"best_individual": best_individual
			}

			plt.plot(fitnesses_history, label=combo_name)

			print(f"Best result: {best_individual}")
			print(f"Crops distribution: {best_individual.genome}")

	plt.title("Comparison of combinations of crossover and mutation operators")
	plt.xlabel("Generation")
	plt.ylabel("Best fitness")
	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	plt.grid(True, alpha=0.3)
	plt.tight_layout()

	plt.savefig(os.path.join(RESULTS_DIR, "crossover_mutation_comparison.png"), dpi=300, bbox_inches="tight")
	plt.show()

	results_df = pd.DataFrame({
		"Combination": list(results.keys()),
		"Best fitness": [results[combo]["best_individual"].fitness for combo in results],
		"Yield": [results[combo]["best_individual"].yield_val for combo in results],
		"Cost": [results[combo]["best_individual"].cost_val for combo in results],
		"Genome": [results[combo]["best_individual"].genome for combo in results]
	}).sort_values("Best fitness", ascending=False)

	results_df.to_csv(os.path.join(RESULTS_DIR, "experiment_results.csv"), index=False)

	print("\n" + "="*60)
	print("THE RESULTS OF THE EXPERIMENT:")
	print("="*60)
	print(results_df.to_string(index=False))

	plt.figure(figsize=(10, 6))
	y_pos = np.arange(len(results_df))
	plt.barh(y_pos, results_df["Best fitness"])
	plt.yticks(y_pos, results_df["Combination"])
	plt.xlabel("Best fitness")
	plt.title("Comparison of the final results by combinations of operators")
	plt.tight_layout()
	plt.savefig(os.path.join(RESULTS_DIR, "final_results_comparison.png"), dpi=300)
	plt.show()


if __name__ == "__main__":
	main_experiment()
