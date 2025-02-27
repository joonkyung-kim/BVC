#!/usr/bin/env python3
"""
BVC Collision Avoidance Differential Drive Simulation

This module implements a collision avoidance simulation using Buffered Voronoi Cells (BVC) for
differential-drive robots. The simulation uses a simple geometric algorithm to determine
collision-free target points within each robot's BVC. The robot motion is updated via a
differential-drive controller, and the robot heading is visualized using a stick (arrow).

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
# Robot and Obstacle Classes
# =============================================================================


class Robot:
    def __init__(self, position, goal, safety_radius, max_speed=0.8, id=None):
        """
        Initialize a differential-drive robot.
        The state consists of [x, y] and orientation theta (radians).
        """
        self.position = np.array(position, dtype=float)  # [x, y]
        self.theta = 0.0  # initial heading (radians)
        self.goal = np.array(goal, dtype=float)
        self.safety_radius = safety_radius
        self.max_speed = max_speed  # maximum linear speed (0.8 m/s)
        self.max_omega = np.pi / 2  # maximum angular speed (pi/2 rad/s)
        self.id = id
        self.trajectory = [self.position.copy()]
        self.orientations = [self.theta]  # record heading history

    def move_diff_drive(self, target_point, dt):
        """
        Move the robot toward a target point using a simple differential-drive controller.
        The desired velocity is computed in the robot's local frame.

        Controller:
            v = error_x / dt      (saturated to max_speed)
            ω = lambda_val * error_y / dt   (saturated to max_omega)
        """
        # Compute error in global coordinates
        error = target_point - self.position
        # Transform error into robot's local frame
        c = math.cos(self.theta)
        s = math.sin(self.theta)
        error_x = c * error[0] + s * error[1]  # forward error
        error_y = -s * error[0] + c * error[1]  # lateral error

        # Compute desired commands
        lambda_val = 2.0  # tuning parameter for angular correction
        v_des = error_x / dt
        omega_des = lambda_val * error_y / dt

        # Saturate commands
        v = np.clip(v_des, 0, self.max_speed)
        omega = np.clip(omega_des, -self.max_omega, self.max_omega)

        # Update state using differential-drive kinematics
        self.position[0] += v * math.cos(self.theta) * dt
        self.position[1] += v * math.sin(self.theta) * dt
        self.theta += omega * dt

        self.trajectory.append(self.position.copy())
        self.orientations.append(self.theta)


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
        Returns (normal, offset) such that any safe point x must satisfy: normal·x <= offset.
        """
        if self.is_point_inside(point):
            dx_left = point[0] - self.xmin
            dx_right = self.xmax - point[0]
            dy_bottom = point[1] - self.ymin
            dy_top = self.ymax - point[1]
            min_dist = min(dx_left, dx_right, dy_bottom, dy_top)
            if min_dist == dx_left:
                normal = np.array([1.0, 0.0])
                offset = np.dot(normal, np.array([self.xmin + safety_radius, point[1]]))
            elif min_dist == dx_right:
                normal = np.array([-1.0, 0.0])
                offset = np.dot(normal, np.array([self.xmax - safety_radius, point[1]]))
            elif min_dist == dy_bottom:
                normal = np.array([0.0, 1.0])
                offset = np.dot(normal, np.array([point[0], self.ymin + safety_radius]))
            else:
                normal = np.array([0.0, -1.0])
                offset = np.dot(normal, np.array([point[0], self.ymax - safety_radius]))
            return (normal, offset)

        closest_point = self.get_closest_point(point)
        dist = np.linalg.norm(closest_point - point)
        if dist <= safety_radius:
            normal = (point - closest_point) / (dist if dist > 1e-10 else 1)
            constraint_point = closest_point + safety_radius * normal
            offset = np.dot(normal, constraint_point)
            return (normal, offset)
        return None


# =============================================================================
# BVC Computation and Utility Functions
# =============================================================================


def compute_boundary_constraints(position, boundary, safety_radius):
    """
    Generate half-space constraints to keep a point inside the boundary.
    The boundary is [xmin, xmax, ymin, ymax].
    """
    xmin, xmax, ymin, ymax = boundary
    constraints = [
        (np.array([-1.0, 0.0]), -(xmin + safety_radius)),
        (np.array([1.0, 0.0]), xmax - safety_radius),
        (np.array([0.0, -1.0]), -(ymin + safety_radius)),
        (np.array([0.0, 1.0]), ymax - safety_radius),
    ]
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
       n^T p <= n^T p_i - 0.5 * ||p_j - p_i|| - r_s,
    where n = (p_j - p_i)/||p_j - p_i||.
    Obstacle and boundary constraints are added.
    """
    constraints = []
    position = robot.position
    base_rs = robot.safety_radius

    # Add constraints from other robots
    for other_robot in all_robots:
        if other_robot.id == robot.id:
            continue
        p_ij = other_robot.position - position
        p_ij_norm = np.linalg.norm(p_ij)
        if p_ij_norm < (robot.safety_radius + other_robot.safety_radius):
            print(f"Warning: Robots {robot.id} and {other_robot.id} are too close!")
            continue
        n = p_ij / p_ij_norm
        if use_right_hand_rule:
            goal_vector = robot.goal - position
            if np.linalg.norm(goal_vector) > 1e-6:
                goal_dir = goal_vector / np.linalg.norm(goal_vector)
                if np.dot(goal_dir, n) > 0.2:
                    right_dir = np.array([goal_dir[1], -goal_dir[0]])
                    side = np.dot(right_dir, n)
                    bias_factor = 0.3
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

        offset = np.dot(n, position) - 0.5 * p_ij_norm - effective_rs
        constraints.append((n, offset))

    # Add obstacle constraints
    if obstacles:
        for obstacle in obstacles:
            obs_constraint = obstacle.get_constraint_for_point(position, base_rs)
            if obs_constraint:
                constraints.append(obs_constraint)

    # Add boundary constraints
    if boundary is not None:
        bnd_constraints = compute_boundary_constraints(position, boundary, base_rs)
        constraints.extend(bnd_constraints)

    return constraints


def is_point_in_bvc(point, constraints):
    """
    Check if a point satisfies all half-space constraints.
    """
    return all(np.dot(normal, point) <= offset for normal, offset in constraints)


def project_point_to_hyperplane(point, hyperplane):
    """
    Project a point onto a hyperplane defined by (normal, offset).
    """
    normal, offset = hyperplane
    normal_unit = normal / np.linalg.norm(normal)
    distance = np.dot(normal_unit, point) - offset
    return point - distance * normal_unit


def find_closest_point_in_bvc(goal, position, constraints):
    """
    Find the closest point to the goal within the BVC using a geometric algorithm.
    """
    if not constraints or is_point_in_bvc(goal, constraints):
        return goal.copy()

    closest_point = None
    min_distance = float("inf")

    # Try projection onto each constraint
    for i, (normal_i, offset_i) in enumerate(constraints):
        projection = project_point_to_hyperplane(goal, (normal_i, offset_i))
        others = [c for j, c in enumerate(constraints) if j != i]
        if is_point_in_bvc(projection, others):
            distance = np.linalg.norm(projection - goal)
            if distance < min_distance:
                min_distance = distance
                closest_point = projection

    # If needed, try intersections
    if closest_point is None:
        for i in range(len(constraints)):
            for j in range(i + 1, len(constraints)):
                n_i, off_i = constraints[i]
                n_j, off_j = constraints[j]
                A = np.vstack([n_i, n_j])
                b = np.array([off_i, off_j])
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

    if closest_point is None:
        direction = goal - position
        if np.linalg.norm(direction) > 1e-6:
            closest_point = position + 0.1 * (direction / np.linalg.norm(direction))
        else:
            closest_point = position.copy()

    return closest_point


def enforce_free_space(point, obstacles, boundary, safety_radius):
    """
    Enforce that a candidate point is inside the boundary and outside any inflated obstacle.
    """
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
                min_dist = min(dx_left, dx_right, dy_bottom, dy_top)
                if min_dist == dx_left:
                    new_point[0] = inflated_xmin
                elif min_dist == dx_right:
                    new_point[0] = inflated_xmax
                elif min_dist == dy_bottom:
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
    At each step, compute the BVC (with obstacles and boundary) and choose a target point in the BVC
    that is as close as possible to the goal. For differential-drive robots, use move_diff_drive.
    """
    for step in range(max_steps):
        all_reached = all(
            np.linalg.norm(robot.position - robot.goal) <= goal_tolerance
            for robot in robots
        )
        if all_reached:
            print(f"All robots reached their goals in {step} steps!")
            return True

        for robot in robots:
            constraints = compute_buffered_voronoi_cell(
                robot, robots, obstacles, use_right_hand_rule, boundary
            )
            target_point = find_closest_point_in_bvc(
                robot.goal, robot.position, constraints
            )
            target_point = enforce_free_space(
                target_point, obstacles, boundary, robot.safety_radius
            )
            robot.move_diff_drive(target_point, dt)

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
        return position + max_radius * np.column_stack((np.cos(angles), np.sin(angles)))
    num_angles = 36
    angles = np.linspace(0, 2 * np.pi, num_angles)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
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
    Animate the simulation of BVC collision avoidance.
    Displays robot positions, trajectories, BVC polygons, and heading sticks.
    """
    # Create simulation copies of robots
    sim_robots = []
    robot_scale = 1.0  # for visualization scaling
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

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=figure_size)
    if boundary:
        xmin, xmax, ymin, ymax = boundary
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    # Create plot elements for robots, headings, trajectories, and BVC polygons
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

    # Plot obstacles
    obstacle_patches = []
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
            obstacle_patches.append(ax.add_patch(rect))

    ax.set_title("BVC Collision Avoidance (Differential Drive with Heading)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, verticalalignment="top")

    # Precompute simulation steps (for animation)
    all_positions = []
    step = 0
    all_reached = False
    while step < max_steps and not all_reached:
        positions = []
        all_reached = all(
            np.linalg.norm(robot.position - robot.goal) <= goal_tolerance
            for robot in sim_robots
        )
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
        all_positions.append((positions, all_reached, step))
        step += 1

    def update(frame):
        positions, reached, current_step = all_positions[frame]
        for i, robot in enumerate(sim_robots):
            # Update robot circle and trajectory
            robot_circles[i].center = robot.trajectory[frame]
            x_data = [pos[0] for pos in robot.trajectory[: frame + 1]]
            y_data = [pos[1] for pos in robot.trajectory[: frame + 1]]
            trajectory_lines[i].set_data(x_data, y_data)
            # Update heading stick based on stored orientation history
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
            # Update BVC polygon
            cons = compute_buffered_voronoi_cell(
                robot, sim_robots, obstacles, use_right_hand_rule, boundary
            )
            poly_points = approximate_bvc_as_polygon(cons, robot.position)
            bvc_polygons[i].set_xy(poly_points)
        status = "COMPLETE" if reached else "IN PROGRESS"
        info_text.set_text(f"Step: {current_step} | Status: {status}")
        return (
            robot_circles + trajectory_lines + heading_lines + bvc_polygons + [info_text]
        )

    anim = animation.FuncAnimation(
        fig, update, frames=len(all_positions), interval=interval, blit=True
    )
    if save_animation:
        anim.save("bvc_collision_avoidance_diff_drive.mp4", writer="ffmpeg", fps=30)
    plt.tight_layout()
    plt.show()


# =============================================================================
# Environment and YAML Loading Functions
# =============================================================================


def load_environment(yaml_file):
    """
    Load environment configuration from a YAML file.
    """
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_environment_from_yaml(yaml_file, robot_radius=0.2, max_speed=0.8):
    """
    Create robots and obstacles from a YAML configuration file.
    """
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
    """
    Run the simulation using the provided YAML environment configuration.
    """
    robots, obstacles, env_size = create_environment_from_yaml(
        yaml_config, robot_radius=0.2 * (40 / 16), max_speed=0.8 * (40 / 16)
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
