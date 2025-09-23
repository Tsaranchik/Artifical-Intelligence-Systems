import numpy as np
import random
import itertools
import time
import os
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Any


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N = 8 # field size
CROPS = [
	("Wheat", 3.4, 120),
	("Corn", 3.6, 160),
	("Soy", 3.5, 140),
	("Barley", 3.2, 110),
	("Potato", 4.0, 310)
]
k = len(CROPS)

POP_SIZE = 120
GENERATIONS = 100
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1
ELITISM = 2
TOURNAMENT_SIZE = 3
RUNS_PER_COMBO = 5

RESULTS_DIR = "ga_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_fields(N: int,
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


def total_yield_and_cost(solution: np.ndarray,
			 yields_matrix: np.ndarray,
			 costs_matrix: np.ndarray
) -> tuple[float, float]:
	n = len(solution)
	idx = (np.arange(n), solution)
	total_yield = float(yields_matrix[idx].sum())
	total_cost = float(costs_matrix[idx].sum())

	return total_yield, total_cost


def compute_theoretical_bounds(yields_matrix: np.ndarray,
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
	if mode not in ["ratio", "weighted"]:
		raise ValueError("Unknown mode name")
	
	total_y, total_c = total_yield_and_cost(solution, yields_matrix, costs_matrix)
	if mode == "ratio":
		if total_c <= 0:
			return -1e9
		return total_y / total_c
	else:
		if bounds is None:
			bounds = compute_theoretical_bounds(yields_matrix, costs_matrix)
		min_y, max_y, min_c, max_c = bounds
		if max_y - min_y == 0:
			ny = 0.0
		else:
			ny = (total_y - min_y) / (max_y - min_y)
		if max_c - min_c == 0:
			nc = 0.0
		else:
			nc = (total_c - min_c) / (max_c - min_c)
		
		return ny - w * nc


def brute_force_best(
		N: int,
		k: int,
		yeilds_matrix: np.ndarray,
		costs_matrix: np.ndarray,
		mode: str = "weighted",
		w: float = 0.5,
		max_enumeration: int = 2000000
) -> Optional[tuple[np.ndarray, float, tuple[float, float]]]:
	total_space = k ** N
	if total_space > max_enumeration:
		print(f"[brute-force] space is too large: {total_space} > {max_enumeration} skip.")
		return None
	best = None
	best_score = -1e18
	bounds = compute_theoretical_bounds(yeilds_matrix, costs_matrix)
	start = time.time()
	for comb in itertools.product(range(k), repeat=N):
		s = fitness_of_solution(np.array(comb), yeilds_matrix, costs_matrix, mode, w, bounds)
		if s > best_score:
			best_score = s
			best = (np.array(comb), s, total_yield_and_cost(np.array(comb), yeilds_matrix, costs_matrix))
	elapsed = time.time() - start
	print(f"[brute-force] complete in {elapsed:.2f}s, cheked {total_space} combinations")
	
	return best


def one_point_crossover(parent1: np.ndarray,
			parent2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
	n = len(parent1)
	if n < 2:
		return parent1.copy(), parent2.copy()
	cp = random.randint(1, n-1)
	child1 = np.concatenate([parent1[:cp], parent2[cp:]])
	child2 = np.concatenate([parent2[:cp], parent1[cp:]])

	return child1, child2


def two_point_crossover(parent1: np.ndarray,
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


def uniform_crossover(parent1: np.ndarray,
		      parent2: np.ndarray,
		      p: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
	n = len(parent1)
	mask = np.random.rand(n) < p
	child1 = np.where(mask, parent1, parent2)
	child2 = np.where(mask, parent2, parent1)
	
	return child1, child2


def random_reset_mutation(child: list[int],
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


def swap_mutation(child: list[int],
		  k: int,
		  mutation_rate: float
) -> list[int]:
	if random.random() < mutation_rate and len(child) >= 2:
		i, j = random.sample(range(len(child)), 2)
		child[i], child[j] = child[j], child[i]
	
	return child


def scramble_mutation(child: list[int],
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


def tournament_selection(pop: list[np.ndarray],
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


def run_ga(
	   yields_matrix: np.ndarray,
	   costs_matrix: np.ndarray,
	   population_size: int = POP_SIZE,
	   generations: int = GENERATIONS,
	   crossover_name: str = "one_point",
	   mutation_name: str = "random_reset",
	   crossover_rate: float = CROSSOVER_RATE,
	   mutation_rate: float = MUTATION_RATE,
	   elitism: int = ELITISM,
	   tournament_size: int = TOURNAMENT_SIZE,
	   mode: str = "weighted",
	   w: float = 0.5
) -> dict[str, Any]:
	N = yields_matrix.shape[0]
	k = yields_matrix.shape[1]
	bounds = compute_theoretical_bounds(yields_matrix, costs_matrix)

	crossovers = {
		"one_point": one_point_crossover,
		"two_point": two_point_crossover,
		"uniform": uniform_crossover
	}
	mutations = {
		"random_reset": random_reset_mutation,
		"swap": swap_mutation,
		"scramble": scramble_mutation
	}
	# if crossover_name not in list(crossovers.keys()) or mutation_rate not in list(mutations.keys()):
	# 	print(type(crossover_name), list(crossovers.keys())[0])
	# 	raise ValueError("Unknown operator name")
	
	crossover_fn = crossovers[crossover_name]
	mutation_fn = mutations[mutation_name]

	pop = np.random.randint(0, k, size=(population_size, N))
	fitnesses = np.array([fitness_of_solution(ind, yields_matrix, costs_matrix, mode=mode, w=w, bounds=bounds) for ind in pop])
	best_history = []
	best_sol = None
	best_fit = -1e18

	for gen in range(generations):
		elite_idx = np.argsort(-fitnesses)[:elitism]
		elites = pop[elite_idx].copy()

		new_pop = elites.tolist()

		while len(new_pop) < population_size:
			parent1 = tournament_selection(pop, fitnesses, tournament_size)
			parent2 = tournament_selection(pop, fitnesses, tournament_size)

			if random.random() < crossover_rate:
				child1, child2 = crossover_fn(parent1, parent2)
			else:
				child1, child2 = parent1.copy(), parent2.copy()
			
			child1 = mutation_fn(child1, k, mutation_rate)
			child2 = mutation_fn(child2, k, mutation_rate)
			new_pop.append(child1)
			if len(new_pop) < population_size:
				new_pop.append(child2)
			pop = np.array(new_pop)
			fitnesses = np.array([fitness_of_solution(ind, yields_matrix, costs_matrix, mode=mode, w=w, bounds=bounds) for ind in pop])
			gen_best_idx = np.argmax(fitnesses)
			gen_best_fit = float(fitnesses[gen_best_idx])
			gen_best_sol = pop[gen_best_idx].copy()
			best_history.append(gen_best_fit)
			if gen_best_fit > best_fit:
				best_fit = gen_best_fit
				best_sol = gen_best_sol.copy()
			
	total_y, total_c = total_yield_and_cost(best_sol, yields_matrix, costs_matrix)
	
	return {
		"best_solution": best_sol,
		"best_fitness": best_fit,
		"best_total_yield": total_y,
		"best_total_cost": total_c,
		"history": best_history
	}


def run_experiments(
		fields_df,
		yields_matrix,
		costs_matrix,
		crop_names,
		crossovers_list = ["one_point", "two_point", "uniform"],
		mutation_list = ["random_reset", "swap", "scramble"],
		runs = RUNS_PER_COMBO,
		population_size = POP_SIZE,
		generations = GENERATIONS,
		mode = "weighted",
		w = 0.5
):
	combos = []
	results = {}
	i = 1
	for co in crossovers_list:
		for mu in mutation_list:
			combos.append((co, mu))
	for co, mu in combos:
		all_hist = []
		all_best_vals = []
		start = time.time()
		for r in range(runs):
			res = run_ga(
				yields_matrix, costs_matrix,
				population_size=population_size,
				generations=generations,
				crossover_name=co,
				mutation_name=mu,
			)
			all_hist.append(res["history"])
			all_best_vals.append(res["best_fitness"])
		elapsed = time.time() - start
		print(f"[exp] {co} + {mu}: runs={runs}, time={elapsed:.1f}s, mean final fitness={np.mean(all_best_vals):.4f}")
		results[(co,mu)] = {
			"histories": np.array(all_hist),
			"finals": np.array(all_best_vals)
		}

		plt.figure(figsize=(10,6))
		for (co, mu), val in results.items():
			mean_hist = val["histories"].mean(axis=0)
			plt.plot(mean_hist, label=f"{co}+{mu}")
		plt.xlabel("Generation")
		plt.ylabel("Best fitness (mean over runs)")
		plt.title("GA convergence: crossover + mutation combinations")
		plt.legend()
		plt.grid(True)
		plt.tight_layout()
		png = os.path.join(RESULTS_DIR, "convergence_combos.png")
		plt.savefig(png)
		print(f"[plot] saved {png}")
		plt.close()

		labels = []
		data = []
		for (co, mu), val in results.items():
			labels.append(f"{co}\n{mu}")
			data.append(val["finals"])
		plt.figure(figsize=(12,6))
		plt.boxplot(data, labels=labels, showmeans=True)
		plt.xticks(rotation=45, ha="right")
		plt.ylabel("Final best fitness")
		plt.title("Final fitness distribution by operator combo")
		plt.tight_layout()
		png2 = os.path.join(RESULTS_DIR, "boxplot_finals.png")
		plt.savefig(png2)
		print(f"[plot] saved {png2}")
		plt.close()

	return results


def main():
	print("Generate data...")
	fields_df, yields_matrix, costs_matrix, crop_names = generate_fields(N, CROPS)
	print(fields_df)
	print("Matrix expected yield (N x k):")
	print(np.round(yields_matrix, 2))
	print("Costs matrix (N x k):")
	print(np.round(costs_matrix, 2))

	bf = brute_force_best(
		N, k,
		yields_matrix, costs_matrix,
		mode="weighted", w=0.5,
		max_enumeration=500000
	)
	if bf is not None:
		sol, score, (ty, tc) = bf
		print("[brute] best score", score, "yield", ty, "cost", tc, "solution", sol)
	
	print("GA (combinations) experiments start...")
	results = run_experiments(
		fields_df, yields_matrix, costs_matrix, crop_names,
	)
	print("Experiments ended. Results in dir: ", RESULTS_DIR)
	print(results)


if __name__ == "__main__":
	main()
	
