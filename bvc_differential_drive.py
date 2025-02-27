"""
Differential Drive Collision Avoidance using Buffered Voronoi Cells (BVC)

This implementation is based on:
  • The 2017 IEEE Robotics and Automation Letters paper:
    "Fast, On-line Collision Avoidance for Dynamic Vehicles Using Buffered Voronoi Cells"
  • The uploaded bvc.py code (which already implements BVC computation,
    geometric planning, obstacle handling, and visualization)
  • Our research notes that add:
      - A QP–based receding horizon controller using CVXPY.
      - Differential–drive (unicycle) dynamics (with state: x, y, theta)
      - A conversion function (int_to_uni) for mapping single–integrator velocity
        into (v, ω) commands.

You can choose between two planners:
  - "qp": uses a QP (via CVXPY) to compute a safe velocity vector.
  - "geometric": uses the closest–point (boundary projection) method.
  
Author: [Your Name]
Date: [Date]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import cvxpy as cp
import math
import argparse
import yaml
import matplotlib.patches as patches
import matplotlib.animation as animation

# ---------------------------------------------------------------------------
# Classes for Robots and Obstacles
# ---------------------------------------------------------------------------


class DifferentialDriveRobot:
    def __init__(self, position, goal, safety_radius, max_speed=1.0, id=None, theta=0.0):
        """
        Differential drive robot with state (x, y, theta)

        Parameters:
        -----------
        position : array-like, [x, y]
            Initial position.
        goal : array-like, [x, y]
            Goal position.
        safety_radius : float
            Safety radius of the robot.
        max_speed : float
            Maximum translational speed.
        id : int
            Robot identifier.
        theta : float
            Initial orientation (radians).
        """
        self.position = np.array(position, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.safety_radius = safety_radius
        self.max_speed = max_speed
        self.id = id
        self.theta = theta  # orientation (radians)
        self.trajectory = [self.position.copy()]
        self.theta_history = [theta]

    def update_state(self, v, omega, dt):
        """
        Update robot state using unicycle kinematics.

        Parameters:
        -----------
        v : float
            Forward (linear) velocity.
        omega : float
            Angular velocity.
        dt : float
            Time step.
        """
        self.position[0] += v * math.cos(self.theta) * dt
        self.position[1] += v * math.sin(self.theta) * dt
        self.theta += omega * dt
        self.trajectory.append(self.position.copy())
        self.theta_history.append(self.theta)


# The Obstacle class is the same as in the uploaded code.
class Obstacle:
    def __init__(self, center, width, height):
        """
        Rectangular obstacle.

        Parameters:
        -----------
        center : array-like, [x, y]
            Center of the obstacle.
        width : float
            Width along the x-axis.
        height : float
            Height along the y-axis.
        """
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
        closest_point = self.get_closest_point(point)
        return np.linalg.norm(closest_point - point)

    def is_point_inside(self, point):
        return self.xmin <= point[0] <= self.xmax and self.ymin <= point[1] <= self.ymax

    def get_constraint_for_point(self, point, safety_radius=0):
        if self.is_point_inside(point):
            # Push point out by using the closest edge.
            dx_left = point[0] - self.xmin
            dx_right = self.xmax - point[0]
            dy_bottom = point[1] - self.ymin
            dy_top = self.ymax - point[1]
            min_dist = min(dx_left, dx_right, dy_bottom, dy_top)
            if min_dist == dx_left:
                normal = np.array([-1.0, 0.0])
                offset = -np.dot(normal, np.array([self.xmin - safety_radius, point[1]]))
            elif min_dist == dx_right:
                normal = np.array([1.0, 0.0])
                offset = -np.dot(normal, np.array([self.xmax + safety_radius, point[1]]))
            elif min_dist == dy_bottom:
                normal = np.array([0.0, -1.0])
                offset = -np.dot(normal, np.array([point[0], self.ymin - safety_radius]))
            else:
                normal = np.array([0.0, 1.0])
                offset = -np.dot(normal, np.array([point[0], self.ymax + safety_radius]))
            return (normal, offset)
        closest_point = self.get_closest_point(point)
        dist = np.linalg.norm(closest_point - point)
        if dist <= safety_radius:
            if np.linalg.norm(closest_point - point) > 1e-10:
                normal = (point - closest_point) / np.linalg.norm(point - closest_point)
            else:
                if closest_point[0] == self.xmin:
                    normal = np.array([-1.0, 0.0])
                elif closest_point[0] == self.xmax:
                    normal = np.array([1.0, 0.0])
                elif closest_point[1] == self.ymin:
                    normal = np.array([0.0, -1.0])
                else:
                    normal = np.array([0.0, 1.0])
            constraint_point = closest_point + safety_radius * normal
            offset = np.dot(normal, constraint_point)
            return (normal, offset)
        return None


# ---------------------------------------------------------------------------
# BVC Helper Functions (from uploaded code)
# ---------------------------------------------------------------------------


def compute_buffered_voronoi_cell(
    robot, all_robots, obstacles=None, use_right_hand_rule=False
):
    """
    Compute BVC as a set of inequality constraints (normal, offset)
    for the given robot relative to others and obstacles.
    """
    constraints = []
    position = robot.position
    safety_radius = robot.safety_radius

    # Optionally, use goal direction for right-hand rule.
    goal_dir = None
    if use_right_hand_rule:
        goal_vector = robot.goal - position
        if np.linalg.norm(goal_vector) > 1e-6:
            goal_dir = goal_vector / np.linalg.norm(goal_vector)

    for other in all_robots:
        if other.id == robot.id:
            continue
        p_ij = other.position - position
        p_ij_norm = np.linalg.norm(p_ij)
        if p_ij_norm < robot.safety_radius + other.safety_radius:
            print(f"Warning: Robots {robot.id} and {other.id} are in collision!")
            continue
        p_ij_unit = p_ij / p_ij_norm
        midpoint = position + 0.5 * p_ij
        # Optionally adjust safety radius based on right-hand rule.
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
        normal = -p_ij_unit  # pointing toward the robot's interior
        offset = np.dot(normal, offset_point)
        constraints.append((normal, offset))

    if obstacles:
        for obs in obstacles:
            obs_constraint = obs.get_constraint_for_point(position, safety_radius)
            if obs_constraint:
                constraints.append(obs_constraint)
    return constraints


def is_point_in_bvc(point, constraints):
    if not constraints:
        return True
    for normal, offset in constraints:
        if np.dot(normal, point) < offset:
            return False
    return True


def project_point_to_hyperplane(point, hyperplane):
    normal, offset = hyperplane
    normal_unit = normal / np.linalg.norm(normal)
    distance = np.dot(normal_unit, point) - offset
    projection = point - distance * normal_unit
    return projection


def find_closest_point_in_bvc(goal, position, constraints):
    if not constraints:
        return goal.copy()
    if is_point_in_bvc(goal, constraints):
        return goal.copy()
    closest_point = None
    min_distance = float("inf")
    for i, (normal_i, offset_i) in enumerate(constraints):
        projection = project_point_to_hyperplane(goal, (normal_i, offset_i))
        if is_point_in_bvc(projection, [c for j, c in enumerate(constraints) if j != i]):
            distance = np.linalg.norm(projection - goal)
            if distance < min_distance:
                min_distance = distance
                closest_point = projection
    if closest_point is None:
        for i in range(len(constraints)):
            for j in range(i + 1, len(constraints)):
                normal_i, offset_i = constraints[i]
                normal_j, offset_j = constraints[j]
                A = np.vstack([normal_i, normal_j])
                b = np.array([offset_i, offset_j])
                try:
                    vertex = np.linalg.solve(A, b)
                    if is_point_in_bvc(
                        vertex, [c for k, c in enumerate(constraints) if k not in [i, j]]
                    ):
                        distance = np.linalg.norm(vertex - goal)
                        if distance < min_distance:
                            min_distance = distance
                            closest_point = vertex
                except np.linalg.LinAlgError:
                    continue
    if closest_point is None:
        direction = goal - position
        distance = np.linalg.norm(direction)
        if distance > 1e-6:
            direction = direction / distance
            closest_point = position + 0.1 * direction
        else:
            closest_point = position.copy()
    return closest_point


# ---------------------------------------------------------------------------
# Conversion: Single-Integrator Velocity -> Differential Drive Commands
# ---------------------------------------------------------------------------


def int_to_uni(dxi, theta, v_max, w_max, K_ang=2.0):
    """
    Convert a desired integrator velocity (dxi) into differential drive commands.

    Parameters:
    -----------
    dxi : numpy.ndarray
        Desired 2D velocity vector in the world frame.
    theta : float
        Current robot orientation (radians).
    v_max : float
        Maximum forward speed.
    w_max : float
        Maximum angular speed.
    K_ang : float
        Gain for angular correction.

    Returns:
    --------
    tuple
        (v, omega): forward velocity and angular velocity.
    """
    norm_dxi = np.linalg.norm(dxi)
    if norm_dxi < 1e-6:
        return 0.0, 0.0
    theta_des = math.atan2(dxi[1], dxi[0])
    # Project desired velocity along current heading:
    v_forward = dxi[0] * math.cos(theta) + dxi[1] * math.sin(theta)
    v_forward = max(-v_max, min(v_max, v_forward))
    theta_err = theta_des - theta
    # Normalize to [-pi, pi]
    theta_err = (theta_err + math.pi) % (2 * math.pi) - math.pi
    omega = K_ang * theta_err
    omega = max(-w_max, min(w_max, omega))
    return v_forward, omega


# ---------------------------------------------------------------------------
# Planning Functions
# ---------------------------------------------------------------------------


def plan_velocity_qp(robot, all_robots, obstacles, dt, v_max):
    """
    QP-based planner: Compute a safe velocity vector such that
    the next position (p + v*dt) stays inside the BVC.
    """
    constraints_bvc = compute_buffered_voronoi_cell(
        robot, all_robots, obstacles, use_right_hand_rule=False
    )
    p = robot.position
    vec_to_goal = robot.goal - p
    if np.linalg.norm(vec_to_goal) > 1e-6:
        v_des = v_max * vec_to_goal / np.linalg.norm(vec_to_goal)
    else:
        v_des = np.zeros(2)
    v = cp.Variable(2)
    objective = cp.Minimize(cp.sum_squares(v - v_des))
    qp_constraints = []
    # Constraint: the next position must satisfy normal·(p + v*dt) >= offset
    for normal, offset in constraints_bvc:
        qp_constraints.append(normal @ (p + v * dt) >= offset)
    # Replace the conic norm constraint with component-wise bounds:
    qp_constraints.append(v[0] <= v_max)
    qp_constraints.append(v[0] >= -v_max)
    qp_constraints.append(v[1] <= v_max)
    qp_constraints.append(v[1] >= -v_max)
    prob = cp.Problem(objective, qp_constraints)
    prob.solve(solver=cp.OSQP, warm_start=True)
    if v.value is not None:
        return np.array(v.value).flatten()
    else:
        return np.zeros(2)


def plan_velocity_geometric(robot, obstacles, all_robots):
    """
    Geometric planner: Use the closest–point projection method to compute
    a target point within the BVC, then return a velocity vector pointing
    from current position toward that target.
    """
    constraints_bvc = compute_buffered_voronoi_cell(
        robot, all_robots, obstacles, use_right_hand_rule=True
    )
    target_point = find_closest_point_in_bvc(robot.goal, robot.position, constraints_bvc)
    direction = target_point - robot.position
    norm_dir = np.linalg.norm(direction)
    if norm_dir < 1e-6:
        return np.zeros(2)
    return (direction / norm_dir) * robot.max_speed


# ---------------------------------------------------------------------------
# Differential Drive Simulation
# ---------------------------------------------------------------------------


def simulate_differential_drive(
    robots,
    obstacles,
    dt=0.1,
    max_steps=1000,
    planner="geometric",
    v_max=1.0,
    w_max=1.0,
    goal_tol=0.1,
):
    """
    Simulate the collision avoidance for differential drive robots.

    Parameters:
    -----------
    robots : list
        List of DifferentialDriveRobot objects.
    obstacles : list
        List of Obstacle objects.
    dt : float
        Simulation time step.
    max_steps : int
        Maximum number of simulation steps.
    planner : str
        "qp" or "geometric" planner selection.
    v_max : float
        Maximum forward speed.
    w_max : float
        Maximum angular speed.
    goal_tol : float
        Distance tolerance to consider a goal reached.

    Returns:
    --------
    list
        Updated list of robots (trajectories updated).
    """
    for step in range(max_steps):
        all_reached = True
        for robot in robots:
            if np.linalg.norm(robot.position - robot.goal) > goal_tol:
                all_reached = False
                break
        if all_reached:
            print(f"All robots reached their goals at step {step}.")
            break

        # For each robot compute desired velocity and update its state.
        for robot in robots:
            if planner == "qp":
                dxi = plan_velocity_qp(robot, robots, obstacles, dt, v_max)
            else:
                dxi = plan_velocity_geometric(robot, obstacles, robots)
            v, omega = int_to_uni(dxi, robot.theta, v_max, w_max)
            robot.update_state(v, omega, dt)
    return robots


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def visualize_trajectories(robots, boundary=None, obstacles=None):
    """
    Plot the trajectories of the robots.
    """
    plt.figure(figsize=(10, 10))
    if boundary:
        xmin, xmax, ymin, ymax = boundary
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
    colors = plt.cm.tab10(np.linspace(0, 1, len(robots)))
    for i, robot in enumerate(robots):
        traj = np.array(robot.trajectory)
        plt.plot(
            traj[:, 0],
            traj[:, 1],
            "-",
            linewidth=2,
            color=colors[i],
            label=f"Robot {robot.id}",
        )
        plt.plot(
            robot.trajectory[0][0],
            robot.trajectory[0][1],
            "o",
            markersize=8,
            color=colors[i],
        )
        plt.plot(robot.goal[0], robot.goal[1], "x", markersize=10, color=colors[i])
        circle = plt.Circle(
            robot.position, robot.safety_radius, fill=True, alpha=0.3, color=colors[i]
        )
        plt.gca().add_patch(circle)
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
            plt.gca().add_patch(rect)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Differential Drive BVC Collision Avoidance Trajectories")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


# ---------------------------------------------------------------------------
# YAML Environment Loader (optional)
# ---------------------------------------------------------------------------


def load_environment(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_environment_from_yaml(yaml_file, robot_radius=0.5, max_speed=1.0):
    config = load_environment(yaml_file)
    robots = []
    for i in range(config["agentNum"]):
        start_point = config["startPoints"][i]
        goal_point = config["goalPoints"][i]
        # Initial orientation can be set to 0 or randomized.
        robot = DifferentialDriveRobot(
            start_point, goal_point, robot_radius, max_speed, id=i, theta=0.0
        )
        robots.append(robot)
    obstacles = []
    for obs_config in config["obstacles"]:
        center = obs_config["center"]
        width = obs_config["width"]
        height = obs_config["height"]
        obstacle = Obstacle(center, width, height)
        obstacles.append(obstacle)
    environment_size = (40, 40)  # Default size; adjust as needed.
    return robots, obstacles, environment_size


# ---------------------------------------------------------------------------
# Main: Run Simulation from YAML or Example
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Differential Drive BVC Collision Avoidance Simulation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (optional)",
    )
    parser.add_argument(
        "--planner",
        type=str,
        default="qp",  # "geometric",
        choices=["qp", "geometric"],
        help="Planner type: 'qp' or 'geometric'",
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Maximum simulation steps"
    )
    parser.add_argument("--dt", type=float, default=0.1, help="Time step")
    args = parser.parse_args()

    if args.config:
        robots, obstacles, env_size = create_environment_from_yaml(
            args.config, robot_radius=0.5, max_speed=1.0
        )
        boundary = (0, env_size[0], 0, env_size[1])
    else:
        # Create an example environment with 4 robots and one obstacle.
        robots = []
        robots.append(
            DifferentialDriveRobot([5, 5], [35, 35], 0.5, max_speed=1.0, id=0, theta=0.0)
        )
        robots.append(
            DifferentialDriveRobot(
                [35, 5], [5, 35], 0.5, max_speed=1.0, id=1, theta=math.pi
            )
        )
        robots.append(
            DifferentialDriveRobot([5, 35], [35, 5], 0.5, max_speed=1.0, id=2, theta=0.0)
        )
        robots.append(
            DifferentialDriveRobot(
                [35, 35], [5, 5], 0.5, max_speed=1.0, id=3, theta=math.pi
            )
        )
        obstacles = [Obstacle([20, 20], 4, 4), Obstacle([30, 30], 4, 4)]
        boundary = (0, 40, 0, 40)

    # Set simulation parameters
    dt = args.dt
    max_steps = args.steps
    planner = args.planner
    v_max = 1.0
    w_max = 1.0

    # Run simulation
    simulate_differential_drive(
        robots, obstacles, dt, max_steps, planner, v_max, w_max, goal_tol=0.5
    )
    visualize_trajectories(robots, boundary, obstacles)


if __name__ == "__main__":
    main()
