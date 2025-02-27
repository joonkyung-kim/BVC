import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import matplotlib.patches as patches
import matplotlib.animation as animation
import yaml
import argparse


class Robot:
    def __init__(self, position, goal, safety_radius, max_speed=1.0, id=None):
        """
        Initialize a robot with single integrator dynamics.
        """
        self.position = np.array(position, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.safety_radius = safety_radius
        self.max_speed = max_speed
        self.id = id
        self.trajectory = [self.position.copy()]

    def move_to_point(self, target_point, dt):
        """
        Move the robot toward a target point with limited speed.
        """
        direction = target_point - self.position
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction = direction / distance
            speed = min(distance / dt, self.max_speed)
            velocity = direction * speed
            self.position = self.position + velocity * dt
        self.trajectory.append(self.position.copy())


class Obstacle:
    def __init__(self, center, width, height):
        """
        Initialize a rectangular obstacle.
        """
        self.center = np.array(center, dtype=float)
        self.width = width
        self.height = height
        self.xmin = self.center[0] - self.width / 2
        self.xmax = self.center[0] + self.width / 2
        self.ymin = self.center[1] - self.height / 2
        self.ymax = self.center[1] + self.height / 2

    def get_closest_point(self, point):
        """
        Get the closest point on the obstacle to the given point.
        """
        closest_x = max(self.xmin, min(point[0], self.xmax))
        closest_y = max(self.ymin, min(point[1], self.ymax))
        return np.array([closest_x, closest_y])

    def distance_to_point(self, point):
        """
        Calculate the distance from the obstacle to the given point.
        """
        closest_point = self.get_closest_point(point)
        return np.linalg.norm(closest_point - point)

    def is_point_inside(self, point):
        """
        Check if a point is inside the obstacle.
        """
        return self.xmin <= point[0] <= self.xmax and self.ymin <= point[1] <= self.ymax

    def get_constraint_for_point(self, point, safety_radius=0):
        """
        Generate a half-space constraint for collision avoidance with the obstacle.
        The constraint is returned in the form (normal, offset) meaning that any
        safe point x must satisfy: normal dot x <= offset.
        """
        # If the point is inside the obstacle, push it out using the nearest edge.
        if self.is_point_inside(point):
            dx_left = point[0] - self.xmin
            dx_right = self.xmax - point[0]
            dy_bottom = point[1] - self.ymin
            dy_top = self.ymax - point[1]
            min_dist = min(dx_left, dx_right, dy_bottom, dy_top)
            if min_dist == dx_left:
                normal = np.array([1.0, 0.0])  # must go right
                offset = np.dot(normal, np.array([self.xmin + safety_radius, point[1]]))
            elif min_dist == dx_right:
                normal = np.array([-1.0, 0.0])  # must go left
                offset = np.dot(normal, np.array([self.xmax - safety_radius, point[1]]))
            elif min_dist == dy_bottom:
                normal = np.array([0.0, 1.0])  # must go up
                offset = np.dot(normal, np.array([point[0], self.ymin + safety_radius]))
            else:  # dy_top
                normal = np.array([0.0, -1.0])  # must go down
                offset = np.dot(normal, np.array([point[0], self.ymax - safety_radius]))
            return (normal, offset)

        # If the point is close to the obstacle, create a constraint to keep a buffer.
        closest_point = self.get_closest_point(point)
        dist = np.linalg.norm(closest_point - point)
        if dist <= safety_radius:
            if dist > 1e-10:
                normal = (point - closest_point) / dist
            else:
                normal = np.array([1.0, 0.0])
            constraint_point = closest_point + safety_radius * normal
            offset = np.dot(normal, constraint_point)
            return (normal, offset)

        return None


def compute_buffered_voronoi_cell(
    robot: Robot,
    all_robots: list[Robot],
    obstacles: list[Obstacle] = None,
    use_right_hand_rule=False,
):
    """
    Compute the BVC constraints for a robot.
    The BVC is defined (for each neighbor j) by the half-space:
         n^T p <= n^T p_i - 0.5*||p_j-p_i|| - r_s
    where n = (p_j-p_i)/||p_j-p_i||.
    Optionally, a right-hand rule may adjust the safety margin.
    Obstacle constraints (in the same half-space form) are also added.
    """
    constraints = []
    position = robot.position
    base_rs = robot.safety_radius

    # Compute constraints from other robots
    for other_robot in all_robots:
        if other_robot.id == robot.id:
            continue

        p_ij = other_robot.position - position
        p_ij_norm = np.linalg.norm(p_ij)
        if p_ij_norm < (robot.safety_radius + other_robot.safety_radius):
            print(f"Warning: Robots {robot.id} and {other_robot.id} are too close!")
            continue

        n = p_ij / p_ij_norm  # unit vector from robot i to j

        # Optionally adjust safety margin with a right-hand rule bias
        if use_right_hand_rule:
            goal_vector = robot.goal - position
            if np.linalg.norm(goal_vector) > 1e-6:
                goal_dir = goal_vector / np.linalg.norm(goal_vector)
                cos_angle = np.dot(goal_dir, n)
                if cos_angle > 0.2:  # if facing roughly the same direction
                    right_dir = np.array([goal_dir[1], -goal_dir[0]])
                    side = np.dot(right_dir, n)
                    bias_factor = 0.3  # adjustable bias factor
                    # Increase the effective safety radius if neighbor lies on the 'undesired' side
                    if side < 0:
                        effective_rs = base_rs * (1 + bias_factor)
                    else:
                        effective_rs = base_rs * (1 - bias_factor * 0.5)
                else:
                    effective_rs = base_rs
            else:
                effective_rs = base_rs
        else:
            effective_rs = base_rs

        # Formulate the half-space:
        # Require: n^T p <= n^T position - 0.5 * p_ij_norm - effective_rs
        offset = np.dot(n, position) - 0.5 * p_ij_norm - effective_rs
        constraints.append((n, offset))

    # Add constraints from obstacles (if any)
    if obstacles:
        for obstacle in obstacles:
            obs_constraint = obstacle.get_constraint_for_point(position, base_rs)
            if obs_constraint:
                constraints.append(obs_constraint)

    return constraints


def is_point_in_bvc(point, constraints):
    """
    Check if a point satisfies all half-space constraints of the BVC.
    Here each constraint is of the form: normal dot x <= offset.
    """
    for normal, offset in constraints:
        if np.dot(normal, point) > offset:  # violation: too high
            return False
    return True


def project_point_to_hyperplane(point, hyperplane):
    """
    Project a point onto a hyperplane defined by (normal, offset):
         normal dot x = offset.
    """
    normal, offset = hyperplane
    normal_unit = normal / np.linalg.norm(normal)
    # Compute the signed distance from point to the hyperplane
    distance = np.dot(normal_unit, point) - offset
    projection = point - distance * normal_unit
    return projection


def find_closest_point_in_bvc(goal, position, constraints):
    """
    Find the closest point to the goal within the BVC using the geometric algorithm.
    If the goal is inside the BVC, return it directly. Otherwise, try projecting
    the goal onto each boundary and then, if needed, compute vertex intersections.
    """
    # If no constraints, BVC is unbounded: return the goal.
    if not constraints:
        return goal.copy()

    # If goal is inside the BVC, no need to modify.
    if is_point_in_bvc(goal, constraints):
        return goal.copy()

    closest_point = None
    min_distance = float("inf")

    # Try projections onto each individual hyperplane
    for i, (normal_i, offset_i) in enumerate(constraints):
        projection = project_point_to_hyperplane(goal, (normal_i, offset_i))
        # Check if the projection satisfies all other constraints
        other_constraints = [c for j, c in enumerate(constraints) if j != i]
        if is_point_in_bvc(projection, other_constraints):
            distance = np.linalg.norm(projection - goal)
            if distance < min_distance:
                min_distance = distance
                closest_point = projection

    # If no valid projection found, compute intersections (vertices) of pairs of constraints.
    if closest_point is None:
        for i in range(len(constraints)):
            for j in range(i + 1, len(constraints)):
                n_i, offset_i = constraints[i]
                n_j, offset_j = constraints[j]
                A = np.vstack([n_i, n_j])
                b = np.array([offset_i, offset_j])
                try:
                    vertex = np.linalg.solve(A, b)
                    remaining = [c for k, c in enumerate(constraints) if k not in [i, j]]
                    if is_point_in_bvc(vertex, remaining):
                        distance = np.linalg.norm(vertex - goal)
                        if distance < min_distance:
                            min_distance = distance
                            closest_point = vertex
                except np.linalg.LinAlgError:
                    continue

    # If still no candidate, take a small step from current position toward goal.
    if closest_point is None:
        direction = goal - position
        if np.linalg.norm(direction) > 1e-6:
            closest_point = position + 0.1 * (direction / np.linalg.norm(direction))
        else:
            closest_point = position.copy()

    return closest_point


def simulate_bvc_collision_avoidance(
    robots: list[Robot],
    dt=0.1,
    max_steps=1000,
    goal_tolerance=0.1,
    use_right_hand_rule=False,
    obstacles=None,
):
    """
    Run the simulation loop: at every time step, compute the BVC (with obstacles),
    then use the geometric algorithm to choose a target point in the BVC that is as
    close as possible to the goal. The robot then moves toward that point.
    """
    for step in range(max_steps):
        all_reached = True
        for robot in robots:
            if np.linalg.norm(robot.position - robot.goal) > goal_tolerance:
                all_reached = False
                break
        if all_reached:
            print(f"All robots reached their goals in {step} steps!")
            return True

        for robot in robots:
            bvc_constraints = compute_buffered_voronoi_cell(
                robot, robots, obstacles, use_right_hand_rule
            )
            target_point = find_closest_point_in_bvc(
                robot.goal, robot.position, bvc_constraints
            )
            robot.move_to_point(target_point, dt)
    print(
        f"Simulation ended after {max_steps} steps. Not all robots reached their goals."
    )
    return False


def approximate_bvc_as_polygon(constraints, position, max_radius=10):
    """
    Approximate the BVC as a polygon for visualization.
    """
    if not constraints:
        angles = np.linspace(0, 2 * np.pi, 20)
        circle_points = position + max_radius * np.column_stack(
            [np.cos(angles), np.sin(angles)]
        )
        return circle_points

    num_angles = 36
    angles = np.linspace(0, 2 * np.pi, num_angles)
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    polygon_points = []
    for direction in directions:
        min_distance = max_radius
        for normal, offset in constraints:
            denom = np.dot(normal, direction)
            if abs(denom) > 1e-10:
                distance = (offset - np.dot(normal, position)) / denom
                if 0 < distance < min_distance:
                    min_distance = distance
        polygon_points.append(position + min_distance * direction)
    return np.array(polygon_points)


def animate_simulation(
    robots,
    dt=0.05,
    max_steps=1000,
    goal_tolerance=0.1,
    figure_size=(10, 10),
    boundary=None,
    interval=50,
    save_animation=False,
    use_right_hand_rule=False,
    obstacles=None,
):
    """
    Animate the simulation of BVC collision avoidance.
    """
    # Make copies of robots for simulation
    sim_robots = []
    for robot in robots:
        new_robot = Robot(
            robot.position.copy(),
            robot.goal.copy(),
            robot.safety_radius,
            robot.max_speed,
            robot.id,
        )
        sim_robots.append(new_robot)

    fig, ax = plt.subplots(figsize=figure_size)
    if boundary:
        xmin, xmax, ymin, ymax = boundary
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    robot_circles = []
    goal_markers = []
    trajectory_lines = []
    bvc_polygons = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(sim_robots)))

    for i, robot in enumerate(sim_robots):
        circle = plt.Circle(
            robot.position, robot.safety_radius, fill=True, alpha=0.5, color=colors[i]
        )
        robot_circles.append(ax.add_patch(circle))
        goal = ax.plot(robot.goal[0], robot.goal[1], "x", markersize=10, color=colors[i])[
            0
        ]
        goal_markers.append(goal)
        (trajectory,) = ax.plot(
            [], [], "-", linewidth=1.5, color=colors[i], label=f"Robot {robot.id}"
        )
        trajectory_lines.append(trajectory)
        polygon = patches.Polygon(
            np.zeros((1, 2)), closed=True, fill=False, edgecolor=colors[i], alpha=0.3
        )
        bvc_polygons.append(ax.add_patch(polygon))

    obstacle_patches = []
    if obstacles:
        for obstacle in obstacles:
            rect = patches.Rectangle(
                (obstacle.xmin, obstacle.ymin),
                obstacle.width,
                obstacle.height,
                linewidth=1,
                edgecolor="k",
                facecolor="gray",
                alpha=0.7,
            )
            obstacle_patches.append(ax.add_patch(rect))

    ax.set_title("Buffered Voronoi Cell Collision Avoidance Animation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, verticalalignment="top")

    all_positions = []
    step = 0
    all_reached = False
    while step < max_steps and not all_reached:
        positions = []
        all_reached = True
        for robot in sim_robots:
            if np.linalg.norm(robot.position - robot.goal) > goal_tolerance:
                all_reached = False
                break
        for robot in sim_robots:
            positions.append(robot.position.copy())
            bvc_constraints = compute_buffered_voronoi_cell(
                robot, sim_robots, obstacles, use_right_hand_rule
            )
            target_point = find_closest_point_in_bvc(
                robot.goal, robot.position, bvc_constraints
            )
            robot.move_to_point(target_point, dt)
        all_positions.append((positions, all_reached, step))
        step += 1

    def update(frame):
        positions, reached, current_step = all_positions[frame]
        for i, robot in enumerate(sim_robots):
            robot_circles[i].center = robot.trajectory[frame]
            x_data = [pos[0] for pos in robot.trajectory[: frame + 1]]
            y_data = [pos[1] for pos in robot.trajectory[: frame + 1]]
            trajectory_lines[i].set_data(x_data, y_data)
            bvc_constraints = compute_buffered_voronoi_cell(
                robot, sim_robots, obstacles, use_right_hand_rule
            )
            polygon_points = approximate_bvc_as_polygon(bvc_constraints, robot.position)
            bvc_polygons[i].set_xy(polygon_points)
        status = "COMPLETE" if reached else "IN PROGRESS"
        info_text.set_text(f"Step: {current_step} | Status: {status}")
        return robot_circles + trajectory_lines + bvc_polygons + [info_text]

    anim = animation.FuncAnimation(
        fig, update, frames=len(all_positions), interval=interval, blit=True
    )
    if save_animation:
        anim.save("bvc_collision_avoidance.mp4", writer="ffmpeg", fps=30)
    plt.tight_layout()
    plt.show()


def load_environment(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_environment_from_yaml(yaml_file, robot_radius=0.2, max_speed=0.8):
    config = None
    if isinstance(yaml_file, str):
        with open(yaml_file, "r") as file:
            config = yaml.safe_load(file)
    else:
        config = yaml_file

    robots = []
    for i in range(config["agentNum"]):
        start_point = config["startPoints"][i]
        goal_point = config["goalPoints"][i]
        robot = Robot(start_point, goal_point, robot_radius, max_speed=max_speed, id=i)
        robots.append(robot)

    obstacles = []
    for obs_config in config["obstacles"]:
        center = obs_config["center"]
        width = obs_config["width"]
        height = obs_config["height"]
        obstacle = Obstacle(center, width, height)
        obstacles.append(obstacle)

    environment_size = (40, 40)
    return robots, obstacles, environment_size


def run_yaml_environment(yaml_config, use_right_hand_rule=True, max_steps=1000, dt=0.05):
    robots, obstacles, env_size = create_environment_from_yaml(yaml_config)
    boundary = (0, env_size[0], 0, env_size[1])
    animate_simulation(
        robots,
        dt=dt,
        max_steps=max_steps,
        boundary=boundary,
        interval=50,
        use_right_hand_rule=use_right_hand_rule,
        obstacles=obstacles,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BVC Collision Avoidance Simulation")
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_wj/rect_maps_wj/RectEnv15/agents20/RectEnv_15_20_2.yaml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()
    yaml_config = load_environment(args.config)
    run_yaml_environment(yaml_config, use_right_hand_rule=True, max_steps=2000)
