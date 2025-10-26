import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def create_fuzzy_system() -> ctrl.ControlSystem:
	battery = ctrl.Antecedent(np.arange(0, 101, 1), 'battery')
	distance = ctrl.Antecedent(np.arange(0, 31, 1), 'distance')
	priority = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'priority')
	suitability = ctrl.Consequent(np.arange(0, 11, 1), 'suitability')

	battery['low'] = fuzz.trimf(battery.universe, [0, 0, 40])
	battery['medium'] = fuzz.trimf(battery.universe, [30, 50, 70])
	battery['high'] = fuzz.trimf(battery.universe, [60, 100, 100])

	distance['near'] = fuzz.trimf(distance.universe, [0, 0, 5])
	distance['medium'] = fuzz.trimf(distance.universe, [3, 8, 13])
	distance['far'] = fuzz.trimf(distance.universe, [10, 20, 30])

	priority['low'] = fuzz.trimf(priority.universe, [0.0, 0.0, 0.4])
	priority['medium'] = fuzz.trimf(priority.universe, [0.3, 0.5, 0.7])
	priority['high'] = fuzz.trimf(priority.universe, [0.6, 1.0, 1.0])

	suitability['low'] = fuzz.trimf(suitability.universe, [0, 0, 4])
	suitability['medium'] = fuzz.trimf(suitability.universe, [3, 5, 7])
	suitability['high'] = fuzz.trimf(suitability.universe, [6, 10, 10])

	rules = [
		ctrl.Rule(battery['high'] & distance['near'] & priority['high'], suitability['high']),
		ctrl.Rule(battery['low'], suitability['low']),
		ctrl.Rule(distance['far'] & priority['low'], suitability['low']),
		ctrl.Rule(battery['medium'] & distance['medium'] & priority['medium'], suitability['medium']),
		ctrl.Rule(battery['high'] & distance['medium'], suitability['medium']),
		ctrl.Rule(battery['medium'] & distance['near'] & priority['high'], suitability['high']),
		ctrl.Rule(battery['high'] & distance['far'] & priority['high'], suitability['medium']),
		ctrl.Rule(~battery['low'] & ~distance['far'] & ~priority['low'], suitability['medium']),
		ctrl.Rule(battery['medium'] | distance['medium'] | priority['medium'], suitability['medium']),
	]
	control_system = ctrl.ControlSystem(rules)

	return control_system


def evaluate_rules(robot, order, dist, control_system):
	sim = ctrl.ControlSystemSimulation(control_system)

	sim.input['battery'] = float(robot.battery)
	d_val = float(min(max(dist, 0.0), 30.0))
	sim.input['distance'] = d_val
	sim.input['priority'] = float(order.priority)
	sim.compute()

	score = 0.0

	if 'suitability' in sim.output:
		score = float(sim.output['suitability'])
	details = {
		'battery': robot.battery,
		'distance': dist,
		'priority': order.priority
	}

	return score, details