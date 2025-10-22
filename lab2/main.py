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


def fuzzy_label(
		value: float,
		x: np.ndarray,
		mfs: dict[str, np.ndarray]
) -> str:
	memberships = {name: fuzz.interp_membership(x, mf, value) for name, mf in mfs.items()}
	return max(memberships, key=memberships.get)

def main() -> None:
	while True:
		pollution_value = float(input("Введите уровень загрязнения воздуха (0-100): "))
		if pollution_value < 0 or pollution_value > 100:
			print("Введены неккоректные значения уровня загрязнения. Попробуйте ещё раз.")
			continue
		noise_value = float(input("Введите уровень шума (0-100): "))
		if noise_value < 0 or noise_value > 100:
			print("Введены неккоректные значения уровня шума. Попробуйте ещё раз.")
			continue
		break

	result = implication(pollution_value, noise_value)
	
	pollution_label = fuzzy_label(pollution_value, pollution, {
		"Чисто": pollution_low,
		"Умеренное загрязнение": pollution_medium,
		"Загрязнено": pollution_high,
		"Сильно загрязнено": pollution_very_high
	})
	noise_label = fuzzy_label(noise_value, noise, {
		"Тихо": noise_low,
		"Средне": noise_medium,
		"Шумно": noise_high,
		"Очень шумно": noise_very_high
	})

	eco_value = fuzz.defuzz(eco_risk, result, 'centroid')
	eco_label = fuzzy_label(eco_value, eco_risk, {
		"Низкая": eco_low,
		"Средняя": eco_medium,
		"Высокая": eco_high,
		"Очень высокая": eco_very_high,
	})

	print(f"\nЗагрязнение ({pollution_value:.1f}): {pollution_label}")
	print(f"Шум ({noise_value:.1f}): {noise_label}")
	print(f"Экологическая опасность ({eco_value:.1f}): {eco_label}")

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