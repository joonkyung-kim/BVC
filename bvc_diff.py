import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import matplotlib.patches as patches
import matplotlib.animation as animation
import yaml
import argparse

GLOBAL_SCALE = 2.5


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
        Differential drive 로봇 초기화

        Parameters:
        -----------
        position : numpy.ndarray
            초기 위치 [x, y]
        goal : numpy.ndarray
            목표 위치 [x, y]
        safety_radius : float
            안전 반경 (여기서는 0.4)
        max_speed : float
            최대 선속도 (기본 0.8)
        max_angular : float
            최대 각속도 (기본 π/2)
        initial_theta : float
            초기 방향 (라디안)
        id : int
            로봇 식별자
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
        목표 지점을 향해 differential drive 방식으로 이동.
        목표점까지의 방향과 현재 방향의 차이에 따라 각속도를 결정하고,
        선속도는 최대 선속도와 cos(각오차)를 곱하여 적용합니다.
        """
        desired_angle = np.arctan2(
            target_point[1] - self.position[1], target_point[0] - self.position[0]
        )
        angle_error = desired_angle - self.theta
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

        # 각속도 제어
        Kp_ang = 4.0
        omega = np.clip(Kp_ang * angle_error, -self.max_angular, self.max_angular)

        distance = np.linalg.norm(target_point - self.position)
        # 선속도는 최대 선속도와 cos(각오차)를 곱해서 결정 (음수이면 0)
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
            closest_point = point.copy()
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


def compute_buffered_voronoi_cell(
    robot: Robot,
    all_robots: list[Robot],
    obstacles: list[Obstacle] = None,
    use_right_hand_rule=False,
):
    """
    Compute the Buffered Voronoi Cell (BVC) for a robot.
    """
    constraints = []
    scale_factor = 2.0  # 1.5
    safety_radius = robot.safety_radius * scale_factor
    position = robot.position
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
                        vertex,
                        [c for k, c in enumerate(constraints) if k != i and k != j],
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


def simulate_bvc_collision_avoidance(
    robots: list[Robot],
    dt=0.1,
    max_steps=1000,
    goal_tolerance=0.1,
    use_right_hand_rule=False,
    obstacles=None,
):
    for step in range(max_steps):
        all_reached = True
        for robot in robots:
            distance_to_goal = np.linalg.norm(robot.position - robot.goal)
            if distance_to_goal > goal_tolerance:
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
    Animate the BVC collision avoidance algorithm.
    로봇의 현재 위치와 heading을 나타내는 막대기도 함께 애니메이션합니다.
    시각화에서는 로봇 반지름을 0.4로 고정합니다.
    """
    viz_radius = 0.5
    sim_robots: list[Robot] = []
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
    heading_lines = []  # heading indicator 선들을 위한 리스트
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
        # heading indicator 초기화: viz_radius 길이의 선분
        x0, y0 = robot.position
        x1 = x0 + viz_radius * np.cos(robot.theta)
        y1 = y0 + viz_radius * np.sin(robot.theta)
        (heading_line,) = ax.plot([x0, x1], [y0, y1], color=colors[i], lw=2)
        heading_lines.append(heading_line)

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
            ax.add_patch(rect)

    ax.set_title("Buffered Voronoi Cell Collision Avoidance Animation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    # ax.legend(loc="upper left")
    ax.set_aspect("equal")
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, verticalalignment="top")
    all_positions = []
    step = 0
    all_reached = False

    while step < max_steps and not all_reached:
        positions = []
        all_reached = True
        for robot in sim_robots:
            distance_to_goal = np.linalg.norm(robot.position - robot.goal)
            if distance_to_goal > goal_tolerance:
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
            # heading indicator 업데이트: 현재 위치와 해당 frame의 theta 사용 (길이 = viz_radius)
            pos = robot.trajectory[frame]
            theta = robot.theta_trajectory[frame]
            x0, y0 = pos
            x1 = x0 + viz_radius * np.cos(theta)
            y1 = y0 + viz_radius * np.sin(theta)
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


def load_environment(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_environment_from_yaml(
    yaml_file, robot_radius=0.5, max_speed=0.8 * GLOBAL_SCALE
):  # -> tuple[list, list, tuple[Literal[40], Literal[40]]]:
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
    parser = argparse.ArgumentParser(description="ORCA Simulation with configurable path")
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_wj/rect_maps_wj/RectEnv15/agents20/RectEnv_15_20_2.yaml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()
    yaml_config = load_environment(args.config)
    run_yaml_environment(yaml_config, use_right_hand_rule=True, max_steps=2000)
