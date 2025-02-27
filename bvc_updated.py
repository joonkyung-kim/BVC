#!/usr/bin/env python3
"""
BVC Collision Avoidance Differential Drive Simulation

This module implements a collision avoidance simulation using Buffered Voronoi Cells (BVC)
for differential-drive robots. The guarantee for collision avoidance in the paper is that if
each robot’s center remains inside its BVC (which is the Voronoi cell retracted by the safety radius),
then the distance between any two centers is always at least 2*rₛ. In our implementation, we add an
extra check after each move: if a robot comes too close to another, we project its state back onto
the BVC. This additional enforcement is in the spirit of the receding–horizon controller in the paper.

Usage:
    python simulation.py --config <path-to-yaml-config>
"""

import numpy as np
import math
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import yaml
import argparse


# =============================================================================
# BVC Computation and Utility Functions
# =============================================================================


def is_point_in_bvc(point, constraints):
    for normal, offset in constraints:
        # If normal·point > offset => outside
        if np.dot(normal, point) > offset:
            return False
    return True


def project_point_to_hyperplane(point, hyperplane):
    """Project a point onto a hyperplane defined by (normal, offset)."""
    n, off = hyperplane
    n_unit = n / np.linalg.norm(n)
    distance = np.dot(n_unit, point) - off
    return point - distance * n_unit


def find_closest_point_in_bvc(goal, position, constraints):
    """
    Find the closest point to the goal within the BVC using a geometric algorithm.
    """
    if not constraints or is_point_in_bvc(goal, constraints):
        return goal.copy()
    closest_point = None
    min_distance = float("inf")
    for i, (n_i, off_i) in enumerate(constraints):
        proj = project_point_to_hyperplane(goal, (n_i, off_i))
        others = [c for j, c in enumerate(constraints) if j != i]
        if is_point_in_bvc(proj, others):
            d = np.linalg.norm(proj - goal)
            if d < min_distance:
                min_distance = d
                closest_point = proj
    if closest_point is None:
        for i in range(len(constraints)):
            for j in range(i + 1, len(constraints)):
                n1, off1 = constraints[i]
                n2, off2 = constraints[j]
                A = np.vstack([n1, n2])
                b = np.array([off1, off2])
                try:
                    vertex = np.linalg.solve(A, b)
                    remaining = [c for k, c in enumerate(constraints) if k not in [i, j]]
                    if is_point_in_bvc(vertex, remaining):
                        d = np.linalg.norm(vertex - goal)
                        if d < min_distance:
                            min_distance = d
                            closest_point = vertex
                except np.linalg.LinAlgError:
                    continue
    if closest_point is None:
        d = np.linalg.norm(goal - position)
        closest_point = position + 0.1 * ((goal - position) / (d if d > 1e-6 else 1))
    return closest_point


def compute_halfplane_intersection(constraints):
    """
    Compute the exact intersection polygon of a set of half-planes.
    Each constraint is (normal, offset): n·x <= offset.
    Returns the vertices of the intersection polygon in counterclockwise order.
    """
    candidates = []
    n_cons = len(constraints)
    for i in range(n_cons):
        for j in range(i + 1, n_cons):
            n1, off1 = constraints[i]
            n2, off2 = constraints[j]
            A = np.array([n1, n2])
            if np.linalg.matrix_rank(A) < 2:
                continue
            try:
                pt = np.linalg.solve(A, np.array([off1, off2]))
            except np.linalg.LinAlgError:
                continue
            if all(np.dot(n, pt) <= off + 1e-6 for n, off in constraints):
                candidates.append(pt)
    if not candidates:
        return None
    candidates = np.array(candidates)
    center = np.mean(candidates, axis=0)
    angles = np.arctan2(candidates[:, 1] - center[1], candidates[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    return candidates[sorted_indices]


def approximate_bvc_as_polygon(constraints, position, max_radius=10):
    """
    Compute the intersection polygon of the half-planes for visualization.
    If the intersection is empty, return a circle approximation.
    """
    poly = compute_halfplane_intersection(constraints)
    if poly is None or len(poly) == 0:
        angles = np.linspace(0, 2 * np.pi, 20)
        return position + max_radius * np.column_stack((np.cos(angles), np.sin(angles)))
    return poly


# =============================================================================
# Robot and Obstacle Classes
# =============================================================================


class Robot:
    def __init__(self, position, goal, safety_radius, max_speed=0.8, id=None):
        self.position = np.array(position, dtype=float)
        self.theta = 0.0
        self.goal = np.array(goal, dtype=float)
        self.safety_radius = safety_radius
        self.max_speed = max_speed
        self.max_omega = np.pi / 2
        self.id = id
        self.trajectory = [self.position.copy()]
        self.orientations = [self.theta]

    def move_diff_drive(self, target_point, dt):
        error = target_point - self.position
        c = math.cos(self.theta)
        s = math.sin(self.theta)
        error_x = c * error[0] + s * error[1]
        error_y = -s * error[0] + c * error[1]
        lambda_val = 2.0
        v_des = error_x / dt
        omega_des = lambda_val * error_y / dt
        v = np.clip(v_des, 0, self.max_speed)
        omega = np.clip(omega_des, -self.max_omega, self.max_omega)
        self.position[0] += v * math.cos(self.theta) * dt
        self.position[1] += v * math.sin(self.theta) * dt
        self.theta += omega * dt
        self.trajectory.append(self.position.copy())
        self.orientations.append(self.theta)


class Obstacle:
    def __init__(self, center, width, height):
        self.center = np.array(center, dtype=float)
        self.width = width
        self.height = height
        self.xmin = self.center[0] - self.width / 2
        self.xmax = self.center[0] + self.width / 2
        self.ymin = self.center[1] - self.height / 2
        self.ymax = self.center[1] + self.height / 2

    def get_closest_point(self, point):
        closest_x = max(self.xmin, min(point[0], self.xmax))
        closest_y = max(self.ymin, min(point[1], self.ymax))
        return np.array([closest_x, closest_y])

    def distance_to_point(self, point):
        return np.linalg.norm(self.get_closest_point(point) - point)

    def is_point_inside(self, point):
        return self.xmin <= point[0] <= self.xmax and self.ymin <= point[1] <= self.ymax


# =============================================================================
# Additional BVC Functions
# =============================================================================


def compute_boundary_constraints(position, boundary, safety_radius):
    xmin, xmax, ymin, ymax = boundary
    return [
        (np.array([-1.0, 0.0]), -(xmin + safety_radius)),
        (np.array([1.0, 0.0]), xmax - safety_radius),
        (np.array([0.0, -1.0]), -(ymin + safety_radius)),
        (np.array([0.0, 1.0]), ymax - safety_radius),
    ]


def compute_obstacle_constraints(robot, obstacle, safety_radius):
    constraints = []
    d = obstacle.distance_to_point(robot.position)
    threshold = 3 * safety_radius
    if d > threshold:
        return constraints
    x, y = robot.position
    if x < obstacle.xmin:
        constraints.append((np.array([1.0, 0.0]), obstacle.xmin - safety_radius))
    elif x > obstacle.xmax:
        constraints.append((np.array([-1.0, 0.0]), -(obstacle.xmax + safety_radius)))
    if y < obstacle.ymin:
        constraints.append((np.array([0.0, 1.0]), obstacle.ymin - safety_radius))
    elif y > obstacle.ymax:
        constraints.append((np.array([0.0, -1.0]), -(obstacle.ymax + safety_radius)))
    return constraints


def compute_buffered_voronoi_cell(
    robot: Robot,
    all_robots: list[Robot],
    obstacles: list[Obstacle] = None,
    use_right_hand_rule=False,
    boundary=None,
):
    """
    Compute the Buffered Voronoi Cell (BVC) constraints for a robot.
    For each neighbor j, the constraint is:
       n^T x <= n^T p_i - 0.5 ||p_j - p_i|| - r_s,
    where n = (p_j - p_i)/||p_j - p_i||.
    Obstacle and boundary constraints are added.
    """
    constraints = []
    pos = robot.position
    base_rs = robot.safety_radius
    for other_robot in all_robots:
        if other_robot.id == robot.id:
            continue
        p_ij = other_robot.position - pos
        p_ij_norm = np.linalg.norm(p_ij)
        # if p_ij_norm < (base_rs + other_robot.safety_radius):
        #     print(
        #         f"Collision detected between robot {robot.id} and robot {other_robot.id}"
        #     )
        #     continue
        n = p_ij / p_ij_norm
        if use_right_hand_rule:
            goal_vector = robot.goal - pos
            if np.linalg.norm(goal_vector) > 1e-6:
                goal_dir = goal_vector / np.linalg.norm(goal_vector)
                if np.dot(goal_dir, n) > 0.2:  # if the robot is facing the goal
                    # Use right-hand rule to determine the side
                    # Compute the right direction
                    right_dir = np.array([goal_dir[1], -goal_dir[0]])
                    # Compute the side of the line
                    side = np.dot(right_dir, n)
                    bias_factor = 0.0  # 0.1  # 0.3
                    effective_rs = (
                        base_rs * (1 + bias_factor)
                        if side < 0
                        else base_rs * (1 - bias_factor * 0.5)
                    )
                else:
                    effective_rs = base_rs
            else:
                effective_rs = base_rs
        else:
            effective_rs = base_rs
        offset = np.dot(n, pos) - 0.5 * p_ij_norm - effective_rs
        constraints.append((n, offset))
    if obstacles:
        for obs in obstacles:
            constraints.extend(compute_obstacle_constraints(robot, obs, base_rs))
    if boundary is not None:
        constraints.extend(compute_boundary_constraints(pos, boundary, base_rs))
    return constraints


# =============================================================================
# Enforcing Free Space and Intersection Functions
# =============================================================================


def enforce_free_space(point, obstacles, boundary, safety_radius):
    new_point = point.copy()
    if boundary is not None:
        xmin, xmax, ymin, ymax = boundary
        new_point[0] = np.clip(new_point[0], xmin + safety_radius, xmax - safety_radius)
        new_point[1] = np.clip(new_point[1], ymin + safety_radius, ymax - safety_radius)
    if obstacles:
        for obs in obstacles:
            inflated_xmin = obs.xmin - safety_radius
            inflated_xmax = obs.xmax + safety_radius
            inflated_ymin = obs.ymin - safety_radius
            inflated_ymax = obs.ymax + safety_radius
            if (inflated_xmin <= new_point[0] <= inflated_xmax) and (
                inflated_ymin <= new_point[1] <= inflated_ymax
            ):
                dx_left = abs(new_point[0] - inflated_xmin)
                dx_right = abs(inflated_xmax - new_point[0])
                dy_bottom = abs(new_point[1] - inflated_ymin)
                dy_top = abs(inflated_ymax - new_point[1])
                m = min(dx_left, dx_right, dy_bottom, dy_top)
                if m == dx_left:
                    new_point[0] = inflated_xmin
                elif m == dx_right:
                    new_point[0] = inflated_xmax
                elif m == dy_bottom:
                    new_point[1] = inflated_ymin
                else:
                    new_point[1] = inflated_ymax
    return new_point


# =============================================================================
# Simulation and Animation Functions
# =============================================================================


def simulate_bvc_collision_avoidance(
    robots: list[Robot],
    dt=0.1,
    max_steps=1000,
    goal_tolerance=0.1,
    use_right_hand_rule=False,
    obstacles=None,
    boundary=None,
):
    """
    Run the simulation loop.
    At each step, compute the BVC and choose a target point that lies within it.
    After moving, if a robot's new position is outside its BVC or too close to another robot,
    project it back onto the BVC boundary.
    """
    for step in range(max_steps):
        if all(
            np.linalg.norm(robot.position - robot.goal) <= goal_tolerance
            for robot in robots
        ):
            print(f"All robots reached their goals in {step} steps!")
            return True

        for robot in robots:
            cons = compute_buffered_voronoi_cell(
                robot, robots, obstacles, use_right_hand_rule, boundary
            )
            target_point = find_closest_point_in_bvc(robot.goal, robot.position, cons)
            target_point = enforce_free_space(
                target_point, obstacles, boundary, robot.safety_radius
            )
            robot.move_diff_drive(target_point, dt)
            # Enforce that robot's new position is in its BVC:
            cons_new = compute_buffered_voronoi_cell(
                robot, robots, obstacles, use_right_hand_rule, boundary
            )

            if not is_point_in_bvc(robot.position, cons_new):
                # poly = compute_halfplane_intersection(cons_new)
                # if poly is None:
                #     poly = approximate_bvc_as_polygon(cons_new, robot.position)
                # dists = np.linalg.norm(poly - robot.position, axis=1)
                # robot.position = poly[np.argmin(dists)]
                # robot.trajectory[-1] = robot.position.copy()
                proj = find_closest_point_in_bvc(robot.position, robot.position, cons_new)
                robot.position = proj
                robot.trajectory[-1] = robot.position.copy()
                # 목표(또는 투영된 점)를 향하도록 heading 재설정
                robot.theta = np.arctan2(
                    robot.goal[1] - robot.position[1], robot.goal[0] - robot.position[0]
                )

            # Additionally, check pairwise distance:
            for other_robot in robots:
                if other_robot.id == robot.id:
                    continue
                if np.linalg.norm(robot.position - other_robot.position) < (
                    robot.safety_radius + other_robot.safety_radius
                ):
                    print(
                        f"Collision detected between robot {robot.id} and robot {other_robot.id}, projecting back."
                    )
                    cons_new = compute_buffered_voronoi_cell(
                        robot, robots, obstacles, use_right_hand_rule, boundary
                    )
                    poly = compute_halfplane_intersection(cons_new)
                    if poly is None:
                        poly = approximate_bvc_as_polygon(cons_new, robot.position)
                    dists = np.linalg.norm(poly - robot.position, axis=1)
                    robot.position = poly[np.argmin(dists)]
                    robot.trajectory[-1] = robot.position.copy()
        # (Optional: record positions for animation)
    print(
        f"Simulation ended after {max_steps} steps. Not all robots reached their goals."
    )
    return False


def animate_simulation(
    robots: list[Robot],
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
    Animate the simulation.
    Displays robot positions, trajectories, computed BVC polygons, and heading sticks.
    """
    sim_robots = []
    robot_scale = 2.0  # 1.2
    for robot in robots:
        new_robot = Robot(
            robot.position.copy(),
            robot.goal.copy(),
            robot.safety_radius * robot_scale,
            robot.max_speed,
            robot.id,
        )
        new_robot.theta = robot.theta
        sim_robots.append(new_robot)

    fig, ax = plt.subplots(figsize=figure_size)
    if boundary:
        xmin, xmax, ymin, ymax = boundary
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    robot_circles = []
    goal_markers = []
    trajectory_lines = []
    heading_lines = []
    bvc_polygons = []
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(sim_robots)))
    for i, robot in enumerate(sim_robots):
        circle = plt.Circle(
            robot.position,
            robot.safety_radius / robot_scale,
            fill=True,
            alpha=0.5,
            color=colors[i],
        )
        robot_circles.append(ax.add_patch(circle))
        goal = ax.plot(robot.goal[0], robot.goal[1], "x", markersize=10, color=colors[i])[
            0
        ]
        goal_markers.append(goal)
        (traj_line,) = ax.plot(
            [], [], "-", linewidth=1.5, color=colors[i], label=f"Robot {robot.id}"
        )
        trajectory_lines.append(traj_line)
        arrow_length = robot.safety_radius / robot_scale * 1.5
        heading_endpoint = robot.position + arrow_length * np.array(
            [math.cos(robot.theta), math.sin(robot.theta)]
        )
        (head_line,) = ax.plot(
            [robot.position[0], heading_endpoint[0]],
            [robot.position[1], heading_endpoint[1]],
            color=colors[i],
            linewidth=2,
        )
        heading_lines.append(head_line)
        poly = patches.Polygon(
            np.zeros((1, 2)), closed=True, fill=False, edgecolor=colors[i], alpha=0.3
        )
        bvc_polygons.append(ax.add_patch(poly))
    if obstacles:
        for obs in obstacles:
            rect = patches.Rectangle(
                (obs.xmin, obs.ymin),
                obs.width,
                obs.height,
                linewidth=1,
                edgecolor="k",
                facecolor="gray",
                alpha=0.7,
            )
            ax.add_patch(rect)
    ax.set_title("BVC Collision Avoidance (Differential Drive with Heading)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, verticalalignment="top")

    all_positions = []
    step = 0
    while step < max_steps:
        if all(
            np.linalg.norm(robot.position - robot.goal) <= goal_tolerance
            for robot in sim_robots
        ):
            print(f"All robots reached their goals in {step} steps!")
            break
        positions = []
        for robot in sim_robots:
            positions.append(robot.position.copy())
            cons = compute_buffered_voronoi_cell(
                robot, sim_robots, obstacles, use_right_hand_rule, boundary
            )
            target_point = find_closest_point_in_bvc(robot.goal, robot.position, cons)
            target_point = enforce_free_space(
                target_point, obstacles, boundary, robot.safety_radius
            )
            robot.move_diff_drive(target_point, dt)
        all_positions.append((positions, step))
        step += 1

    def update(frame):
        positions, current_step = all_positions[frame]
        for i, robot in enumerate(sim_robots):
            robot_circles[i].center = robot.trajectory[frame]
            x_data = [pos[0] for pos in robot.trajectory[: frame + 1]]
            y_data = [pos[1] for pos in robot.trajectory[: frame + 1]]
            trajectory_lines[i].set_data(x_data, y_data)
            current_pos = robot.trajectory[frame]
            current_theta = robot.orientations[frame]
            arrow_length = robot.safety_radius / robot_scale * 1.5
            heading_endpoint = current_pos + arrow_length * np.array(
                [math.cos(current_theta), math.sin(current_theta)]
            )
            heading_lines[i].set_data(
                [current_pos[0], heading_endpoint[0]],
                [current_pos[1], heading_endpoint[1]],
            )
            cons = compute_buffered_voronoi_cell(
                robot, sim_robots, obstacles, use_right_hand_rule, boundary
            )
            poly_points = compute_halfplane_intersection(cons)
            if poly_points is None:
                poly_points = approximate_bvc_as_polygon(cons, robot.position)
            bvc_polygons[i].set_xy(poly_points)
        info_text.set_text(f"Step: {current_step}")
        return (
            robot_circles + trajectory_lines + heading_lines + bvc_polygons + [info_text]
        )

    anim = animation.FuncAnimation(
        fig, update, frames=len(all_positions), interval=interval, blit=True
    )
    plt.tight_layout()
    plt.show()


# =============================================================================
# Environment and YAML Loading Functions
# =============================================================================


def load_environment(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_environment_from_yaml(yaml_file, robot_radius=0.2, max_speed=0.8):
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
    robots, obstacles, env_size = create_environment_from_yaml(
        yaml_config, robot_radius=0.2 * (40 / 16), max_speed=0.8
    )
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


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BVC Collision Avoidance Differential Drive Simulation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_wj/rect_maps_wj/RectEnv15/agents20/RectEnv_15_20_2.yaml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()
    yaml_config = load_environment(args.config)
    run_yaml_environment(yaml_config, use_right_hand_rule=True, max_steps=2000)
