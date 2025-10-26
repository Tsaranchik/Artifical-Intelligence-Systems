import random
from entities.robot import Robot
from entities.order import Order
from simulation.ontology import warehouse_ontology
from logic.assigment import choose_robot_for_order
from logic.fuzzy_system import create_fuzzy_system
import config


class WarehouseSimulator:
	def __init__(self, ontology: dict, num_robots: int = 4) -> None:
		self.ontology = ontology or warehouse_ontology
		self.robots: list[Robot] = []
		self.orders: list[Order] = []
		self.step = 0
		self.log = []
		self.last_order_step = 0
		self.fuzzy_system = create_fuzzy_system()

		charging_station = self.ontology["charging_station"]["pos"]
		for i in range(num_robots):
			start_pos = (
				charging_station[0] + random.uniform(-1,1),
				charging_station[1] + random.uniform(-1,1)
			)
			r = Robot(id=i+1, pos=start_pos, 
	     			  battery=random.uniform(30, 100),
				  speed=config.ROBOT_SPEED
				)
			self.robots.append(r)
		
	def spawn_random_order(self) -> None:
		shelf = random.choice(self.ontology['shelves'])
		item = random.choice(shelf['items'])
		order = Order(
			id = 0 + len(self.orders) + 1,
			item=item,
			from_shelf=shelf["id"],
			to_pos=(self.ontology['size'][0]-1, random.uniform(2, self.ontology['size'][1]-2)),
			priority=random.uniform(0.1, 1),
			created_step=self.step
		)
		self.orders.append(order)
		print(f"[Step {self.step}] New order {order.id} item={order.item} from {order.from_shelf} priority={order.priority:.2f}")
	
	def step_simulation(self) -> None:
		self.step += 1

		active_orders = len([o for o in self.orders if not o.delivered])
		adaptive_probability = max(
			config.MIN_ORDER_PROBABILITY,
			config.BASE_ORDER_SPAWN_PROBABILITY - active_orders * config.ORDER_PROBABILITY_DECAY
		)
		should_spawn_order = (
			active_orders < config.MAX_ACTIVE_ORDERS and
			self.step - self.last_order_step >= config.ORDER_COOLDOWN and
			random.random() < adaptive_probability
		)

		if should_spawn_order:
			self.spawn_random_order()
			self.last_order_step = self.step
		
		for r in self.robots:
			if r.task is not None:
				self.process_robot_task(r)
		
		free_robots = [r for r in self.robots if r.task is None and r.status != "charging"]
		pending_orders = [o for o in self.orders if not o.picked and not o.delivered]

		if free_robots and pending_orders:
			for robot in free_robots:
				best_assigmnet = None
				best_score = 0

				for order in pending_orders:
					order_assigned = any(
						r.task and r.task.get("order_id") == order.id
						for r in self.robots
					)
					if order_assigned:
						continue

					assigment = choose_robot_for_order([robot], order, self.ontology, self.fuzzy_system)
					if assigment:
						_, score, details = assigment
						if score > best_score and score > config.MIN_ASSIGNMENT_SCORE:
							best_score = score
							best_assigmnet = (order, score, details)
				
				if best_assigmnet:
					order, score, details = best_assigmnet
					robot.task = {
						"order_id": order.id,
						"stage": "to_shelf",
						"shelf": order.from_shelf,
						"to_pos": order.to_pos
					}
					robot.status = "moving"
					print(f"[Step {self.step}] Assigned Robot {robot.id} to Order {order.id} (score={score:.2f})")
					self.log.append((self.step, robot.id, order.id, score))
		
		for r in self.robots:
			if r.task is None:
				if r.battery < config.BATTERY_CRITICAL:
					r.task = {"order_id": None, "stage": "to_charge"}
					r.status = "moving"
					print(f"[Step {self.step}] Robot {r.id} low battery ({r.battery:.1f}%) -> heading to charge")
				else:
					r.status = "idle"
					r.step_battery_use()
	
	def process_robot_task(self, r: Robot) -> None:
		if r.task.get("stage") == "to_charge":
			target = self.ontology["charging_station"]["pos"]
			if r.distance_to(target) > config.THRESHOLD_DISTANCE:
				r.move_towards(target)
				r.status = "moving"
			else:
				r.status = "charging"
				r.battery = min(100, r.battery + config.BATTERY_CHARGE_RATE)
				if r.battery >= 100:
					r.battery = 100
					r.task = None
					r.status = "idle"
					print(f"[Step {self.step}] Robot {r.id} fully charged and ready")
			return
	
		order = next((o for o in self.orders if o.id == r.task["order_id"]), None)
		if order is None:
			r.task = None
			r.status = "idle"
			r.step_battery_use()
			print(f"[Step {self.step}] WARNING: Order not found for Robot {r.id}, freeing robot")
			return

		if order.delivered:
			r.task = None
			r.status = "idle"
			r.step_battery_use()
			print(f"[Step {self.step}] Order {order.id} already delivered, freeing Robot {r.id}")

		if r.task.get("stage") == "to_shelf":
			shelf_pos = next(s["pos"] for s in self.ontology["shelves"] if s["id"] == r.task["shelf"])
			if r.distance_to(shelf_pos) > config.THRESHOLD_DISTANCE:
				r.move_towards(shelf_pos)
				r.status = "moving"
			else:
				if not order.picked:
					order.picked = True
					r.task["stage"] = "to_dropoff"
					r.status = "busy"
					print(f"[Step {self.step}] Robot {r.id} picked order {order.id} at {r.pos}")
				else:
					r.task["stage"] = "to_dropoff"
					r.status = "moving"
		
		elif r.task.get("stage") == "to_dropoff":
			if r.distance_to(order.to_pos) > config.THRESHOLD_DISTANCE:
				r.move_towards(order.to_pos)
				r.status = "moving"
			else:
				order.delivered = True
				r.task = None
				r.status = "idle"
				print(f"[Step {self.step}] Robot {r.id} delivered order {order.id} to {order.to_pos}")
		
		r.step_battery_use()

		if r.battery < config.BATTERY_LOW and r.task is None:
			r.task = {"order_id": None, "stage": "to_charge"}
					
					
		

		
			
	def get_robot_positions(self):
		return [r.pos for r in self.robots]
	
	def get_robot_states(self):
		return [(r.id, r.pos, r.battery, r.status, r.task) for r in self.robots]
	
	def get_active_orders(self):
		return [o for o in self.orders if not o.delivered]
				
