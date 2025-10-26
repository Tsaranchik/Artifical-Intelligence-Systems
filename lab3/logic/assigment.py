from typing import Optional
from entities.robot import Robot
from entities.order import Order
from logic.fuzzy_system import evaluate_rules


def choose_robot_for_order(
		robots: list[Robot],
		order: Order,
		ontology,
		control_system
) -> Optional[tuple[Robot, float, dict]]:
	candidates = []
	shelf_pos = next(s['pos'] for s in ontology['shelves'] if s['id'] == order.from_shelf)

	for r in robots:
		if r.status == "charging":
			continue
	
		dist = r.distance_to(shelf_pos)
		suitability_score, details = evaluate_rules(r, order, dist, control_system)
		penalty = 0.0

		if r.task is not None and r.task.get("order_id") != order.id:
			penalty = 2.5
		
		candidates.append((r, suitability_score - penalty, details))

	if not candidates:
		return None
	
	candidates.sort(key=lambda x: x[1], reverse=True)
	best = candidates[0]
	if best[1] <= 0.5:
		return None
	
	return best[0], best[1], best[2]