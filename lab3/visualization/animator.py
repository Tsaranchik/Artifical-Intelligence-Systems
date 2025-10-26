import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from simulation.warehouse import WarehouseSimulator
from simulation.ontology import warehouse_ontology


def run_visual_simulation(frames=240, interval=600, num_robots=4):
	sim = WarehouseSimulator(warehouse_ontology, num_robots=num_robots)

	fig, ax = plt.subplots(figsize=(10, 6))

	for _ in range(2):
		sim.spawn_random_order()
	
	ax.set_xlim(-1, warehouse_ontology["size"][0]+1)
	ax.set_ylim(-1, warehouse_ontology["size"][1]+1)
	ax.set_title("Warehouse simulator")
	ax.set_xlabel("X")
	ax.set_ylabel("Y")

	shelf_position = [s["pos"] for s in warehouse_ontology["shelves"]]
	ax.scatter([p[0] for p in shelf_position], [p[1] for p in shelf_position])
	cs_pos = warehouse_ontology["charging_station"]["pos"]
	ax.scatter([cs_pos[0]], [cs_pos[1]], s=250, marker='D')

	robot_scatter = ax.scatter([], [], s=120)
	robot_texts = [ax.text(0,0,"") for _ in range(num_robots)]
	order_scatter = ax.scatter([], [], s=100, marker='x')
	legend_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, verticalalignment='top')


	def init():
		robot_scatter.set_offsets(np.empty((0, 2)))
		order_scatter.set_offsets(np.empty((0, 2)))
		for t in robot_texts:
			t.set_text("")
		legend_text.set_text("")
		return robot_scatter, order_scatter, *robot_texts, legend_text

	def update(frame):
		sim.step_simulation()

		positions = sim.get_robot_positions()
		robot_scatter.set_offsets(positions)

		for i, r in enumerate(sim.robots):
			robot_texts[i].set_position((r.pos[0]+0.3, r.pos[1]+0.3))
			robot_texts[i].set_text(f"R{r.id} B:{r.battery:.0f}%\n{r.status}")
		
		active_orders = sim.get_active_orders()
		drops = [o.to_pos for o in active_orders if not o.delivered]

		if len(drops) > 0:
			order_scatter.set_offsets(drops)
		else:
			order_scatter.set_offsets(np.empty((0, 2)))

		info_lines = [f"Step: {sim.step} Active orders: {len(active_orders)}"]
		for r in sim.robots:
			info_lines.append(f"R{r.id}: {r.status} Bat:{r.battery:.0f}%")
		legend_text.set_text("\n".join(info_lines))

		return robot_scatter, order_scatter, *robot_texts, legend_text
	
	ani = animation.FuncAnimation(fig, update, init_func=init, frames=frames, interval=interval, blit=False, repeat=False)
	ani.save("simulation.gif", writer="pillow", fps=10)

	print("\n\n=== Simulation log summary (assignments) ===")
	for rec in sim.log:
		print(f"Step {rec[0]}: Robot {rec[1]} -> Order {rec[2]} (score {rec[3]:.2f})")

