from dataclasses import dataclass


@dataclass
class Order:
	id: int
	item: str
	from_shelf: str
	to_pos: tuple[float, float]
	priority: float
	created_step: int
	picked: bool = False
	delivered: bool = False