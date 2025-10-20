import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


pollution = np.arrange(0, 101, 1)
noise = np.arrange(0, 101, 1)
eco_risk = np.arrange(0, 101, 1)

pollution_low = fuzz.trapmf(pollution, [0, 0, 10, 30])
pollution_medium = fuzz.trimf(pollution, [20, 40, 60])
pollution_high = fuzz.trimf(pollution, [50, 70, 90])
pollution_very_high = fuzz.trapmf(pollution, [80, 90, 100, 100])

noise_low = fuzz.trapmf(noise, [0, 0, 10, 30])
noise_medium = fuzz.trimf(noise, [20, 40, 60])
noise_high = fuzz.trimf(noise, [50, 70, 90])
noise_very_high = fuzz.trapmf(noise, [80, 90, 100, 100])

eco_low = fuzz.trapmf(eco_risk, [0, 0, 10, 30])
eco_medium = fuzz.trimf(eco_risk, [20, 40, 60])
eco_high = fuzz.trimf(eco_risk, [50, 70, 90])
eco_very_high = fuzz.trapmf(eco_risk, [80, 90, 100, 100])

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