import numpy as np
import skfuzzy as fuzz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def create_mfs(x: np.ndarray, name: str) -> dict[str, np.ndarray]:
	mfs = {}
	for level in ["low", "medium", "high", "very_high"]:
		while True:
			try:
				params = list(map(float, input(f"Введите 4 числа для {name} ({level}): ").split()))
				if len(params) != 4:
					raise ValueError("Должно быть ровно 4 числа.")
				if not all(0 <= p <= 100 for p in params):
					raise ValueError("Все числа должны быть от 0 до 100.")
				if not (params[0] <= params[1] <= params[2] <= params[3]):
					raise ValueError("Числа должны быть упорядочены: a <= b <= c <= d")
				
				mfs[level] = fuzz.trapmf(x, params)
				break
			except ValueError as e:
				print(f"Невверный ввод: {e}. попробуйте снова.")
			
	return mfs 


def implication(
		pollution_val: int, noise_val: int,
		pollution_mfs: dict, noise_mfs: dict,
		eco_mfs: dict
) -> np.ndarray:
	mu_pollution = {
		lvl: fuzz.interp_membership(np.arange(0, 101, 1), mf, pollution_val)
		for lvl, mf in pollution_mfs.items()
	}
	mu_noise = {
		lvl: fuzz.interp_membership(np.arange(0, 101, 1), mf, noise_val)
		for lvl, mf in noise_mfs.items()
	}

	eco_activations = []

	rules = {
		('low', 'low'): 'low',
		('low', 'medium'): 'medium',
		('low', 'high'): 'medium',
		('low', 'very_high'): 'high',

		('medium', 'low'): 'medium',
		('medium', 'medium'): 'medium',
		('medium', 'high'): 'high',
		('medium', 'very_high'): 'high',

		('high', 'low'): 'medium',
		('high', 'medium'): 'high',
		('high', 'high'): 'high',
		('high', 'very_high'): 'very_high',

		('very_high', 'low'): 'high',
		('very_high', 'medium'): 'high',
		('very_high', 'high'): 'very_high',
		('very_high', 'very_high'): 'very_high',
	}

	x_eco = np.arange(0, 101)

	for (p_lvl, n_lvl), eco_lvl in rules.items():
		mu_rule = np.fmin(mu_pollution[p_lvl], mu_noise[n_lvl])
		eco_activations.append(np.fmin(mu_rule, eco_mfs[eco_lvl]))
	
	aggregated = np.fmax.reduce(eco_activations)

	return aggregated


def fuzzy_label(
		value: float,
		x: np.ndarray,
		mfs: dict[str, np.ndarray]
) -> str:
	memberships = {name: fuzz.interp_membership(x, mf, value) for name, mf in mfs.items()}
	return max(memberships, key=memberships.get)

def main() -> None:
	x = np.arange(0, 101, 1)
	print("=== Ввод функций принадлежности для загрязнения воздуха ===")
	polltion_mfs = create_mfs(x, "Загрязнение")

	print("\n=== Ввод функцй принадлежности для уровня шума ===")
	noise_mfs = create_mfs(x, "Шум")

	print("\n=== Ввод функций принадлежности для экологической опасности ===")
	eco_mfs = create_mfs(x, "Экологическая опасность")

	while True:
		try:
			pollution_value = float(input("Введите уровень загрязнения воздуха (0-100): "))
			noise_value = float(input("Введите уровень шума (0-100): "))
			if not (0 <= pollution_value <= 100) or not (0 <= noise_value <= 100):
				raise ValueError
			break
		except ValueError:
			print("Неверный ввод. Введите числа от 0 до 100.")


	result = implication(pollution_value, noise_value, polltion_mfs, noise_mfs, eco_mfs)
	
	eco_value = fuzz.defuzz(x, result, 'centroid')
	pollution_label = fuzzy_label(pollution_value, x, polltion_mfs)
	noise_label = fuzzy_label(noise_value, x, noise_mfs)
	eco_label = fuzzy_label(eco_value, x, eco_mfs)

	print(f"\nЗагрязнение ({pollution_value:.1f}): {pollution_label}")
	print(f"Шум ({noise_value:.1f}): {noise_label}")
	print(f"Экологическая опасность ({eco_value:.1f}): {eco_label}")

	plt.figure(figsize=(8, 4))
	for lvl, mf in eco_mfs.items():
		plt.plot(x, mf, linestyle='--', label=lvl)
	plt.fill_between(x, 0, result, color='orange', alpha=0.6, label="Импликация")
	plt.title(f'Результат импликации: экологическая опаность\n(Загрязнение {pollution_value}, Шум: {noise_value})')
	plt.xlabel('Экологическая опасность')
	plt.ylabel('Степень принадлежности')
	plt.legend()
	plt.savefig(f"eco_result.png")


if __name__ == '__main__':
	main()