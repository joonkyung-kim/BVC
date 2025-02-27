import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import matplotlib.patches as patches
import matplotlib.animation as animation
import yaml
import argparse
import math


# ---------------------------------------------------------------------------
# Differential Drive Robot Class
# ---------------------------------------------------------------------------
class DifferentialDriveRobot:
    def __init__(self, position, goal, safety_radius, max_speed=1.0, id=None, theta=0.0):
        """
        Initialize a differential drive robot with unicycle dynamics.

        Parameters:
        -----------
        position : numpy.ndarray
            Initial position [x, y].
        goal : numpy.ndarray
            Goal position [x, y].
        safety_radius : float
            Safety radius.
        max_speed : float
            Maximum translational speed.
        id : int
            Robot identifier.
        theta : float
            Initial heading (radians).
        """
        self.position = np.array(position, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.safety_radius = safety_radius
        self.max_speed = max_speed
        self.id = id
        self.theta = theta  # robot heading (radians)
        self.trajectory = [self.position.copy()]
        self.theta_history = [self.theta]

    def move_to_target(self, target_point, dt, v_max, w_max):
        """
        Update the robot's state using differential drive dynamics.
        Instead of moving directly toward the target (as in single integrator),
        compute the desired integrator velocity and convert it to (v, ω) commands.

        Parameters:
        -----------
        target_point : numpy.ndarray
            The target point (e.g. closest point in the BVC toward the goal).
        dt : float
            Time step.
        v_max : float
            Maximum forward speed.
        w_max : float
            Maximum angular speed.
        """
        # Compute desired integrator velocity as (target - current)/dt.
        desired_vel = (target_point - self.position) / dt
        # Convert desired velocity (in world frame) to unicycle commands.
        v, omega = int_to_uni(desired_vel, self.theta, v_max, w_max)
        # Update state using simple Euler integration of unicycle kinematics.
        self.position[0] += v * math.cos(self.theta) * dt
        self.position[1] += v * math.sin(self.theta) * dt
        self.theta += omega * dt
        # Store history.
        self.trajectory.append(self.position.copy())
        self.theta_history.append(self.theta)


# ---------------------------------------------------------------------------
# Obstacle Class (unchanged)
# ---------------------------------------------------------------------------
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
        closest_point = self.get_closest_point(point)
        return np.linalg.norm(closest_point - point)

    def is_point_inside(self, point):
        return self.xmin <= point[0] <= self.xmax and self.ymin <= point[1] <= self.ymax

    def get_constraint_for_point(self, point, safety_radius=0):
        if self.is_point_inside(point):
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
# BVC Helper Functions (from your code)
# ---------------------------------------------------------------------------
def compute_buffered_voronoi_cell(
    robot, all_robots, obstacles=None, use_right_hand_rule=False
):
    constraints = []
    position = robot.position
    safety_radius = robot.safety_radius

    goal_dir = None
    if use_right_hand_rule:
        goal_vector = robot.goal - position
        if np.linalg.norm(goal_vector) > 1e-6:
            goal_dir = goal_vector / np.linalg.norm(goal_vector)

    for other_robot in all_robots:
        if other_robot.id == robot.id:
            continue
        p_ij = other_robot.position - position
        p_ij_norm = np.linalg.norm(p_ij)
        if p_ij_norm < robot.safety_radius + other_robot.safety_radius:
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

    if obstacles:
        for obstacle in obstacles:
            obstacle_constraint = obstacle.get_constraint_for_point(
                position, safety_radius
            )
            if obstacle_constraint:
                constraints.append(obstacle_constraint)

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
# New: int_to_uni() Function for Differential Drive Conversion
# ---------------------------------------------------------------------------
def int_to_uni(dxi, theta, v_max, w_max, K_ang=2.0):
    """
    Convert a desired single-integrator velocity (dxi) into unicycle
    commands (forward velocity v and angular velocity ω) for a differential drive robot.

    Parameters:
    -----------
    dxi : numpy.ndarray
        Desired 2D velocity vector (in world frame).
    theta : float
        Current robot heading (radians).
    v_max : float
        Maximum forward speed.
    w_max : float
        Maximum angular speed.
    K_ang : float, optional
        Proportional gain for angular error.

    Returns:
    --------
    tuple
        (v, omega): forward velocity and angular velocity.
    """
    norm_dxi = np.linalg.norm(dxi)
    if norm_dxi < 1e-6:
        return 0.0, 0.0
    # Desired heading angle (in world frame)
    theta_des = math.atan2(dxi[1], dxi[0])
    # Forward speed is the projection of dxi onto the current heading
    v_forward = dxi[0] * math.cos(theta) + dxi[1] * math.sin(theta)
    # Clip the forward speed to v_max
    v_forward = max(-v_max, min(v_max, v_forward))
    # Angular error (normalized to [-pi, pi])
    theta_err = theta_des - theta
    theta_err = (theta_err + math.pi) % (2 * math.pi) - math.pi
    omega = K_ang * theta_err
    omega = max(-w_max, min(w_max, omega))
    return v_forward, omega


# ---------------------------------------------------------------------------
# Simulation Function for Differential Drive Robots
# ---------------------------------------------------------------------------
def simulate_differential_drive(
    robots,
    obstacles,
    dt=0.1,
    max_steps=1000,
    goal_tolerance=0.1,
    v_max=1.0,
    w_max=1.0,
    use_right_hand_rule=False,
):
    """
    Simulate the BVC collision avoidance algorithm for differential drive robots.

    For each robot, compute its BVC, determine the closest point in the BVC to its goal,
    then compute a desired velocity (as in a single integrator model). Convert that velocity
    to unicycle commands (v, ω) using int_to_uni() and update the robot's state accordingly.

    Parameters:
    -----------
    robots : list
        List of DifferentialDriveRobot objects.
    obstacles : list
        List of Obstacle objects.
    dt : float
        Time step.
    max_steps : int
        Maximum simulation steps.
    goal_tolerance : float
        Distance tolerance to consider the goal reached.
    v_max : float
        Maximum forward speed.
    w_max : float
        Maximum angular speed.
    use_right_hand_rule : bool
        Whether to apply the right-hand rule.

    Returns:
    --------
    list
        Updated list of robots (with trajectories).
    """
    for step in range(max_steps):
        all_reached = True
        for robot in robots:
            if np.linalg.norm(robot.position - robot.goal) > goal_tolerance:
                all_reached = False
                break
        if all_reached:
            print(f"All robots reached their goals in {step} steps!")
            break

        for robot in robots:
            bvc_constraints = compute_buffered_voronoi_cell(
                robot, robots, obstacles, use_right_hand_rule
            )
            target_point = find_closest_point_in_bvc(
                robot.goal, robot.position, bvc_constraints
            )

            # robot.move_to_target(target_point, dt, v_max, w_max)
            # Check for collisions with obstacles

            # Compute desired integrator velocity (dxi) as (target - current)/dt
            dxi = (target_point - robot.position) / dt
            # Convert to unicycle commands
            v, omega = int_to_uni(dxi, robot.theta, v_max, w_max)
            # Update the robot's state using differential drive dynamics
            robot.position[0] += v * math.cos(robot.theta) * dt
            robot.position[1] += v * math.sin(robot.theta) * dt
            robot.theta += omega * dt
            robot.trajectory.append(robot.position.copy())
            robot.theta_history.append(robot.theta)

    return robots


# ---------------------------------------------------------------------------
# Visualization Functions (unchanged)
# ---------------------------------------------------------------------------
def visualize_simulation(robots, figure_size=(10, 10), boundary=None, obstacles=None):
    plt.figure(figsize=figure_size)
    if boundary:
        xmin, xmax, ymin, ymax = boundary
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
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
        circle = plt.Circle(robot.position, robot.safety_radius, fill=True, alpha=0.3)
        plt.gca().add_patch(circle)
    plt.grid(True)
    plt.legend()
    plt.title("Differential Drive BVC Collision Avoidance Simulation")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.show()


def approximate_bvc_as_polygon(constraints, position, max_radius=10):
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


# ---------------------------------------------------------------------------
# Environment Loader Functions (optional)
# ---------------------------------------------------------------------------
def load_environment(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_environment_from_yaml(yaml_config, robot_radius=0.2, max_speed=0.8):
    # config = load_environment(yaml_file)
    config = yaml_config
    robots = []
    for i in range(config["agentNum"]):
        start_point = config["startPoints"][i]
        goal_point = config["goalPoints"][i]
        robot = DifferentialDriveRobot(
            start_point, goal_point, robot_radius, max_speed=max_speed, id=i, theta=0.0
        )
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


def run_yaml_environment(
    yaml_config,
    use_right_hand_rule=True,
    max_steps=1000,
    dt=0.05,
    v_max=1.0,
    w_max=np.pi / 2,
):
    robots, obstacles, env_size = create_environment_from_yaml(yaml_config)
    boundary = (0, env_size[0], 0, env_size[1])
    simulate_differential_drive(
        robots,
        obstacles,
        dt,
        max_steps,
        goal_tolerance=0.5,
        v_max=v_max,
        w_max=w_max,
        use_right_hand_rule=use_right_hand_rule,
    )
    # visualize_simulation(robots, boundary, obstacles)
    visualize_simulation(
        robots, figure_size=(10, 10), boundary=boundary, obstacles=obstacles
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Differential Drive BVC Collision Avoidance Simulation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_wj/rect_maps_wj/RectEnv15/agents20/RectEnv_15_20_2.yaml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()
    yaml_config = load_environment(args.config)

    max_linear_vel = 0.8
    max_angular_vel = np.pi / 2

    run_yaml_environment(
        yaml_config,
        use_right_hand_rule=True,
        max_steps=2000,
        dt=0.05,
        v_max=max_linear_vel,  # 1.0,
        w_max=max_angular_vel,  # 1.0,
    )
