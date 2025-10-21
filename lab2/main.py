import numpy as np
import skfuzzy as fuzz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


pollution = np.arange(0, 101, 1)
noise = np.arange(0, 101, 1)
eco_risk = np.arange(0, 101, 1)

pollution_low = fuzz.trapmf(pollution, [0, 0, 15, 35])
pollution_medium = fuzz.trapmf(pollution, [25, 40, 55, 70])
pollution_high = fuzz.trapmf(pollution, [60, 75, 85, 95])
pollution_very_high = fuzz.trapmf(pollution, [85, 90, 100, 100])

noise_low = fuzz.trapmf(pollution, [0, 0, 15, 35])
noise_medium = fuzz.trapmf(pollution, [25, 40, 55, 70])
noise_high = fuzz.trapmf(pollution, [60, 75, 85, 95])
noise_very_high = fuzz.trapmf(pollution, [85, 90, 100, 100])

eco_low = fuzz.trapmf(pollution, [0, 0, 15, 35])
eco_medium = fuzz.trapmf(pollution, [25, 40, 55, 70])
eco_high = fuzz.trapmf(pollution, [60, 75, 85, 95])
eco_very_high = fuzz.trapmf(pollution, [85, 90, 100, 100])

def implication(pollution_val: int, noise_val: int) -> np.ndarray:
	mu_pollution = {
		"low": fuzz.interp_membership(pollution, pollution_low, pollution_val),
		"medium": fuzz.interp_membership(pollution, pollution_medium, pollution_val),
		"high": fuzz.interp_membership(pollution, pollution_high, pollution_val),
		"very_high": fuzz.interp_membership(pollution, pollution_very_high, pollution_val),
	}

	mu_noise = {
		"low": fuzz.interp_membership(noise, noise_low, noise_val),
		"medium": fuzz.interp_membership(noise, noise_medium, noise_val),
		"high": fuzz.interp_membership(noise, noise_high, noise_val),
		"very_high": fuzz.interp_membership(noise, noise_very_high, noise_val),
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

	for (p_lvl, n_lvl), eco_lvl in rules.items():
		mu_rule = np.fmin(mu_pollution[p_lvl], mu_noise[n_lvl])
		if eco_lvl == 'low':
			eco_activations.append(np.fmin(mu_rule, eco_low))
		elif eco_lvl == 'medium':
			eco_activations.append(np.fmin(mu_rule, eco_medium))
		elif eco_lvl == 'high':
			eco_activations.append(np.fmin(mu_rule, eco_high))
		elif eco_lvl == 'very_high':
			eco_activations.append(np.fmin(mu_rule, eco_very_high))
	
	aggregated = np.fmax.reduce(eco_activations)

	return aggregated


def main() -> None:
	pollution_value = float(input("Введите уровень загрязнения воздуха (0-100): "))
	noise_value = float(input("Введите уровень шума (0-100): "))

	result = implication(pollution_value, noise_value)

	plt.figure(figsize=(8, 4))
	plt.plot(eco_risk, eco_low, 'b', linestyle='--', label='eco_low')
	plt.plot(eco_risk, eco_medium, 'g', linestyle='--', label='eco_medium')
	plt.plot(eco_risk, eco_high, 'y', linestyle='--', label='eco_high')
	plt.plot(eco_risk, eco_very_high, 'r', linestyle='--', label='eco_very_high')
	plt.fill_between(eco_risk, 0, result, color='orange', alpha=0.6, label='Импликация')
	plt.title(f'Результат импликации: экологическая опаность\n(Загрязнение {pollution_value}, Шум: {noise_value})')
	plt.xlabel('Экологическая опасность')
	plt.ylabel('Степень принадлежности')
	plt.legend()
	plt.savefig(f"eco_result.png")


if __name__ == '__main__':
	main()