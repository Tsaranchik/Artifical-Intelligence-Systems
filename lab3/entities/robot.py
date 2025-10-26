from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class Robot:
	id: int
	pos: tuple[float, float]
	battery: float
	speed: float = 1.0
	task: Optional[dict] = None
	status: str = "idle"

	def distance_to(self, point: tuple[float, float]) -> float:
		return math.hypot(self.pos[0] - point[0], self.pos[1] - point[1])

	def move_towards(self, target: tuple[float, float]) -> None:
		if self.pos == target:
			return
		dx = target[0] - self.pos[0]
		dy = target[1] - self.pos[1]
		dist = math.hypot(dx, dy)
		if dist <= self.speed or dist == 0:
			self.pos = (target[0], target[1])
		else:
			self.pos = (
				self.pos[0] + dx / dist * self.speed,
				self.pos[1] + dy / dist * self.speed
			)
	
	def step_battery_use(self) -> None:
		if self.status in ("moving", "busy"):
			self.battery -= 0.8
		elif self.status == "idle":
			self.battery -= 0.03
		if self.battery < 0:
			self.battery = 0