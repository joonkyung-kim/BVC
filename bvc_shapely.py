#!/usr/bin/env python3
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import matplotlib.patches as patches
import matplotlib.animation as animation
import yaml
import argparse
from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points

# =============================================================================
# Global Variables
# =============================================================================
GLOBAL_SCALE = 2.5


# =============================================================================
# Class Definitions
# =============================================================================
class Robot:
    def __init__(
        self,
        position,
        goal,
        safety_radius,
        max_speed=0.8,
        id=None,
        max_angular=np.pi / 2,
        initial_theta=0.0,
    ):
        """
        Differential drive robot initialization.

        Parameters:
        -----------
        position : numpy.ndarray
            Initial position [x, y]
        goal : numpy.ndarray
            Goal position [x, y]
        safety_radius : float
            Safety radius (e.g., 0.4)
        max_speed : float
            Maximum linear speed
        max_angular : float
            Maximum angular speed
        initial_theta : float
            Initial orientation (radians)
        id : int
            Robot identifier
        """
        self.position = np.array(position, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.safety_radius = safety_radius
        self.max_speed = max_speed
        self.max_angular = max_angular
        self.theta = initial_theta
        self.id = id
        self.trajectory = [self.position.copy()]
        self.theta_trajectory = [self.theta]

    def move_to_point(self, target_point, dt):
        """
        Move the robot toward the target point using differential drive kinematics.
        The linear speed is modulated by the cosine of the heading error.
        """
        desired_angle = np.arctan2(
            target_point[1] - self.position[1], target_point[0] - self.position[0]
        )
        angle_error = desired_angle - self.theta
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

        # Angular control
        Kp_ang = 10.0
        omega = np.clip(Kp_ang * angle_error, -self.max_angular, self.max_angular)

        distance = np.linalg.norm(target_point - self.position)
        # Linear speed is maximum speed times cos(angle_error) (nonnegative)
        v = self.max_speed * max(0, np.cos(angle_error))
        v = min(v, distance / dt)

        self.position[0] += v * np.cos(self.theta) * dt
        self.position[1] += v * np.sin(self.theta) * dt
        self.theta += omega * dt
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

        self.trajectory.append(self.position.copy())
        self.theta_trajectory.append(self.theta)


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
        # Create a shapely polygon for the rectangular obstacle.
        self.polygon = Polygon(
            [
                (self.xmin, self.ymin),
                (self.xmin, self.ymax),
                (self.xmax, self.ymax),
                (self.xmax, self.ymin),
            ]
        )

    def get_closest_point(self, point):
        """
        Return the closest point on the obstacle's boundary to the given point.
        """
        p = Point(point)
        proj_dist = self.polygon.exterior.project(p)
        closest = self.polygon.exterior.interpolate(proj_dist)
        return np.array([closest.x, closest.y])

    def distance_to_point(self, point):
        closest_point = self.get_closest_point(point)
        return np.linalg.norm(closest_point - point)

    def is_point_inside(self, point):
        return self.polygon.contains(Point(point))

    def get_constraint_for_point(self, point, safety_radius=0):
        """
        Compute a half-plane constraint so that any point satisfying
        np.dot(normal, p) >= offset lies outside the obstacle plus safety margin.
        """
        p = np.array(point, dtype=float)
        p_shapely = Point(p)
        distance = p_shapely.distance(self.polygon)
        if self.polygon.contains(p_shapely) or distance < safety_radius:
            closest = self.get_closest_point(p)
            diff = p - closest
            norm = np.linalg.norm(diff)
            if norm < 1e-6:
                normal = np.array([1.0, 0.0])
            else:
                normal = diff / norm
            constraint_point = closest + safety_radius * normal
            offset = np.dot(normal, constraint_point)
            return (normal, offset)
        return None


class CircleObstacle:
    def __init__(self, center, radius):
        """
        Initialize a circular obstacle.
        """
        self.center = np.array(center, dtype=float)
        self.radius = radius
        # Represent the circle as a shapely polygon.
        self.polygon = Point(self.center).buffer(self.radius, resolution=64)

    def get_closest_point(self, point):
        """
        Return the closest point on the obstacle's boundary to the given point.
        """
        p = Point(point)
        proj_dist = self.polygon.exterior.project(p)
        closest = self.polygon.exterior.interpolate(proj_dist)
        return np.array([closest.x, closest.y])

    def distance_to_point(self, point):
        closest = self.get_closest_point(point)
        return np.linalg.norm(closest - point)

    def is_point_inside(self, point):
        return self.polygon.contains(Point(point))

    def get_constraint_for_point(self, point, safety_radius=0):
        """
        Compute a half-plane constraint for the circular obstacle.
        """
        p = np.array(point, dtype=float)
        p_shapely = Point(p)
        distance = p_shapely.distance(self.polygon)
        if self.polygon.contains(p_shapely) or distance < safety_radius:
            closest = self.get_closest_point(p)
            diff = p - closest
            norm = np.linalg.norm(diff)
            if norm < 1e-6:
                normal = np.array([1.0, 0.0])
            else:
                normal = diff / norm
            constraint_point = closest + safety_radius * normal
            offset = np.dot(normal, constraint_point)
            return (normal, offset)
        return None


# =============================================================================
# Helper Functions (BVC and Projection)
# =============================================================================
def compute_buffered_voronoi_cell(
    robot: Robot, all_robots: list, obstacles: list = None, use_right_hand_rule=False
):
    """
    Compute the Buffered Voronoi Cell (BVC) for a robot.
    It computes half-plane constraints from neighboring robots and obstacles.
    """
    constraints = []
    scale_factor = 2.0
    safety_radius = robot.safety_radius * scale_factor
    position = robot.position
    goal_dir = None
    if use_right_hand_rule:
        goal_vector = robot.goal - position
        if np.linalg.norm(goal_vector) > 1e-6:
            goal_dir = goal_vector / np.linalg.norm(goal_vector)
    # Constraints from neighboring robots
    for other_robot in all_robots:
        if other_robot.id == robot.id:
            continue
        p_ij = other_robot.position - position
        p_ij_norm = np.linalg.norm(p_ij)
        if p_ij_norm < (robot.safety_radius + other_robot.safety_radius):
            print(f"Warning: Robots {robot.id} and {other_robot.id} are in collision!")
            continue
        p_ij_unit = p_ij / p_ij_norm
        midpoint = position + 0.5 * p_ij
        if use_right_hand_rule and goal_dir is not None:
            cos_angle = np.dot(goal_dir, p_ij_unit)
            if cos_angle > 0.2:
                right_dir = np.array([goal_dir[1], -goal_dir[0]])
                side_preference = np.dot(right_dir, p_ij_unit)
                bias_factor = 0.3
                if side_preference < 0:
                    safety_radius_adjusted = safety_radius * (1 + bias_factor)
                else:
                    safety_radius_adjusted = safety_radius * (1 - bias_factor * 0.5)
            else:
                safety_radius_adjusted = safety_radius
        else:
            safety_radius_adjusted = safety_radius
        offset_point = midpoint - safety_radius_adjusted * p_ij_unit
        normal = -p_ij_unit
        offset = np.dot(normal, offset_point)
        constraints.append((normal, offset))
    # Add obstacle constraints (as before)
    if obstacles:
        for obstacle in obstacles:
            obs_constraint = obstacle.get_constraint_for_point(position, safety_radius)
            if obs_constraint:
                constraints.append(obs_constraint)
    return constraints


def approximate_bvc_as_polygon(constraints, position, max_radius=10):
    """
    Given the half-plane constraints and a center, approximate the BVC as a polygon.
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
            normal_dot_dir = np.dot(normal, direction)
            if abs(normal_dot_dir) > 1e-10:
                distance = (offset - np.dot(normal, position)) / normal_dot_dir
                if 0 < distance < min_distance:
                    min_distance = distance
        polygon_points.append(position + min_distance * direction)
    return np.array(polygon_points)


# New functions to compute BVC polygon using shapely and subtract obstacles.
def compute_BVC_polygon(
    robot: Robot, all_robots: list, obstacles: list = None, use_right_hand_rule=False
):
    """
    Compute the BVC as a shapely Polygon.
    First, compute the approximate polygon from neighbor constraints.
    Then subtract the obstacles from it.
    """
    constraints = compute_buffered_voronoi_cell(
        robot, all_robots, obstacles, use_right_hand_rule
    )
    poly_points = approximate_bvc_as_polygon(constraints, robot.position)
    bvc_poly = Polygon(poly_points)
    # Subtract each obstacle polygon from the BVC
    if obstacles:
        for obs in obstacles:
            bvc_poly = bvc_poly.difference(obs.polygon)
    return bvc_poly


def find_closest_point_in_bvc_shapely(
    robot: Robot, all_robots: list, obstacles: list = None, use_right_hand_rule=False
):
    """
    Use shapely to compute the final safe region (BVC) and return the closest point
    on it to the robot's goal. If the computed BVC is empty, fallback to a small step
    toward the goal.
    """
    bvc_poly = compute_BVC_polygon(robot, all_robots, obstacles, use_right_hand_rule)
    goal_pt = Point(robot.goal)
    if not bvc_poly.is_empty:
        if bvc_poly.contains(goal_pt):
            return robot.goal.copy()
        else:
            # Compute nearest point on the non-empty polygon
            nearest = nearest_points(goal_pt, bvc_poly)[1]
            return np.array([nearest.x, nearest.y])
    else:
        # Fallback: if safe region is empty, move a small step toward the goal.
        print(f"Warning: BVC polygon is empty for robot {robot.id}. Using fallback.")
        direction = robot.goal - robot.position
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction = direction / norm
            return robot.position + 0.1 * direction
        else:
            return robot.position.copy()


# =============================================================================
# Simulation Functions
# =============================================================================
def simulate_bvc_collision_avoidance(
    robots: list,
    dt=0.1,
    max_steps=1000,
    goal_tolerance=0.1,
    use_right_hand_rule=False,
    obstacles=None,
):
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
            # Use the shapely-based nearest point to incorporate obstacles in the safe region.
            target_point = find_closest_point_in_bvc_shapely(
                robot, robots, obstacles, use_right_hand_rule
            )
            robot.move_to_point(target_point, dt)
    print(
        f"Simulation ended after {max_steps} steps. Not all robots reached their goals."
    )
    return False


def visualize_simulation(robots, figure_size=(10, 10), boundary=None, obstacles=None):
    """
    Visualize the simulation results.
    """
    viz_radius = 0.4
    plt.figure(figsize=figure_size)
    if boundary:
        xmin, xmax, ymin, ymax = boundary
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
    if obstacles:
        for obstacle in obstacles:
            # Draw circles for circular obstacles; rectangles for rectangular obstacles.
            if hasattr(obstacle, "radius"):
                circ_patch = patches.Circle(
                    obstacle.center,
                    obstacle.radius,
                    linewidth=1,
                    edgecolor="k",
                    facecolor="gray",
                    alpha=0.7,
                )
                plt.gca().add_patch(circ_patch)
            else:
                rect = patches.Rectangle(
                    (obstacle.xmin, obstacle.ymin),
                    obstacle.width,
                    obstacle.height,
                    linewidth=1,
                    edgecolor="k",
                    facecolor="gray",
                    alpha=0.7,
                )
                plt.gca().add_patch(rect)
    for robot in robots:
        trajectory = np.array(robot.trajectory)
        plt.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            "-",
            linewidth=2,
            label=f"Robot {robot.id}",
        )
        plt.plot(trajectory[0, 0], trajectory[0, 1], "o", markersize=10)
        plt.plot(robot.goal[0], robot.goal[1], "x", markersize=10)
        circle = plt.Circle(robot.position, viz_radius, fill=True, alpha=0.3)
        plt.gca().add_patch(circle)
        x_start, y_start = robot.position
        x_end = x_start + viz_radius * np.cos(robot.theta)
        y_end = y_start + viz_radius * np.sin(robot.theta)
        plt.plot([x_start, x_end], [y_start, y_end], color="k", lw=2)
    plt.grid(True)
    plt.legend()
    plt.title("Buffered Voronoi Cell Collision Avoidance Simulation")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.show()


def animate_simulation(
    robots: list,
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
    Animate the BVC collision avoidance algorithm.
    Visualizes robot positions, trajectories, and heading indicators.
    """
    viz_radius = 0.5
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
    heading_lines = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(sim_robots)))
    for i, robot in enumerate(sim_robots):
        circle = plt.Circle(
            robot.position, viz_radius, fill=True, alpha=0.5, color=colors[i]
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
        x0, y0 = robot.position
        x1 = x0 + viz_radius * np.cos(robot.theta)
        y1 = y0 + viz_radius * np.sin(robot.theta)
        (heading_line,) = ax.plot([x0, x1], [y0, y1], color=colors[i], lw=2)
        heading_lines.append(heading_line)
    if obstacles:
        for obstacle in obstacles:
            if hasattr(obstacle, "radius"):
                circ_patch = patches.Circle(
                    obstacle.center,
                    obstacle.radius,
                    linewidth=1,
                    edgecolor="k",
                    facecolor="gray",
                    alpha=0.7,
                )
                ax.add_patch(circ_patch)
            else:
                rect = patches.Rectangle(
                    (obstacle.xmin, obstacle.ymin),
                    obstacle.width,
                    obstacle.height,
                    linewidth=1,
                    edgecolor="k",
                    facecolor="gray",
                    alpha=0.7,
                )
                ax.add_patch(rect)
    ax.set_title("Buffered Voronoi Cell Collision Avoidance Animation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.set_aspect("equal")
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, verticalalignment="top")
    all_positions = []
    step = 0
    all_reached = False
    while step < max_steps and not all_reached:
        print(f"Step: {step}")
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
            # Use the shapely-based approach to determine the target point
            target_point = find_closest_point_in_bvc_shapely(
                robot, sim_robots, obstacles, use_right_hand_rule
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
            poly_points = approximate_bvc_as_polygon(bvc_constraints, robot.position)
            # Build a polygon and subtract obstacles for visualization.
            bvc_poly = Polygon(poly_points)
            if obstacles:
                for obs in obstacles:
                    bvc_poly = bvc_poly.difference(obs.polygon)
            if bvc_poly.is_empty:
                pts = np.array([])
            else:
                pts = np.array(bvc_poly.exterior.coords)
            bvc_polygons[i].set_xy(pts)
            pos_frame = robot.trajectory[frame]
            theta_frame = robot.theta_trajectory[frame]
            x0, y0 = pos_frame
            x1 = x0 + viz_radius * np.cos(theta_frame)
            y1 = y0 + viz_radius * np.sin(theta_frame)
            heading_lines[i].set_data([x0, x1], [y0, y1])
        status = "COMPLETE" if reached else "IN PROGRESS"
        info_text.set_text(f"Step: {current_step} | Status: {status}")
        return (
            robot_circles + trajectory_lines + bvc_polygons + heading_lines + [info_text]
        )

    anim = animation.FuncAnimation(
        fig, update, frames=len(all_positions), interval=interval, blit=True
    )
    if save_animation:
        anim.save("bvc_collision_avoidance.mp4", writer="ffmpeg", fps=30)
    plt.tight_layout()
    plt.show()


# =============================================================================
# YAML Environment Functions
# =============================================================================
def load_environment(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_environment_from_yaml(
    yaml_file, robot_radius=0.5, max_speed=0.8 * GLOBAL_SCALE
):
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
    for obs_config in config.get("obstacles", []):
        if "radius" in obs_config:
            center = obs_config["center"]
            radius = obs_config["radius"]
            obstacle = CircleObstacle(center, radius)
        elif "width" in obs_config and "height" in obs_config:
            center = obs_config["center"]
            width = obs_config["width"]
            height = obs_config["height"]
            obstacle = Obstacle(center, width, height)
        else:
            continue
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


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BVC Collision Avoidance Simulation with Obstacles"
    )
    # Define the obstacle type here: "circle", "rectangle", or "free"
    obstacle_type = "circle"  # Options: "circle", "rectangle", "free"
    num_robots = 20
    map_density = 15
    instance_id = 2
    if obstacle_type == "circle":
        map_path = f"benchmark_wj/circle_maps_wj/CircleEnv{map_density}/agents{num_robots}/CircleEnv_{map_density}_{num_robots}_{instance_id}.yaml"
    elif obstacle_type == "rectangle":
        map_path = f"benchmark_wj/rect_maps_wj/RectEnv{map_density}/agents{num_robots}/RectEnv_{map_density}_{num_robots}_{instance_id}.yaml"
    else:
        map_path = f"benchmark_wj/Free_maps_wj/agents{num_robots}/Free_0_{num_robots}_{instance_id}.yaml"
    print(f"Loading map from: {map_path}")
    parser.add_argument(
        "--config", type=str, default=map_path, help="Path to the YAML configuration file"
    )
    args = parser.parse_args()
    yaml_config = load_environment(args.config)
    run_yaml_environment(yaml_config, use_right_hand_rule=True, max_steps=2000)
