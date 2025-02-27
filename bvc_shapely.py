import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import matplotlib.patches as patches
import matplotlib.animation as animation
import yaml
import argparse

# NEW IMPORTS FOR SHAPELY
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

GLOBAL_SCALE = 2.5


def force_inside_free_space(point, free_space):
    """
    If 'point' is not inside the free_space polygon, project it onto free_space.
    """
    p = Point(point)
    if free_space.contains(p):
        return point
    # If free_space is a MultiPolygon, take the largest.
    if free_space.geom_type == "MultiPolygon":
        poly = max(free_space.geoms, key=lambda poly: poly.area)
    else:
        poly = free_space
    proj_distance = poly.exterior.project(p)
    closest_point = poly.exterior.interpolate(proj_distance)
    return np.array([closest_point.x, closest_point.y])


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

    def move_to_point(self, target_point, dt, free_space=None):
        """
        Moves the robot toward target_point using differential drive control.
        If free_space is provided, the new position is forced to lie within free_space.
        """
        desired_angle = np.arctan2(
            target_point[1] - self.position[1], target_point[0] - self.position[0]
        )
        angle_error = desired_angle - self.theta
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

        # Angular speed control
        Kp_ang = 10.0
        omega = np.clip(Kp_ang * angle_error, -self.max_angular, self.max_angular)

        distance = np.linalg.norm(target_point - self.position)
        # Linear speed: maximum speed scaled by cos(angle_error)
        v = self.max_speed * max(0, np.cos(angle_error))
        v = min(v, distance / dt)

        new_position = (
            self.position
            + np.array([v * np.cos(self.theta), v * np.sin(self.theta)]) * dt
        )

        # Force new_position into free_space if provided.
        if free_space is not None:
            new_position = force_inside_free_space(new_position, free_space)

        self.position = new_position
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
        # (Not used in the new approach.)
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
    robot: Robot, all_robots: list, obstacles: list = None, use_right_hand_rule=False
):
    """
    Compute the Buffered Voronoi Cell (BVC) for a robot from its neighbors.
    Obstacles are handled separately.
    """
    constraints = []
    # Using scale_factor 1.0 so that the BVC reflects just neighbor separation.
    scale_factor = 1.0
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


def compute_free_space_polygon(environment_size, obstacles, inflation):
    """
    Compute the configuration-space free region by buffering obstacles by 'inflation'.
    """
    env_width, env_height = environment_size
    env_poly = Polygon([(0, 0), (0, env_height), (env_width, env_height), (env_width, 0)])
    obstacle_polys = []
    for obs in obstacles:
        obs_poly = Polygon(
            [
                (obs.xmin, obs.ymin),
                (obs.xmin, obs.ymax),
                (obs.xmax, obs.ymax),
                (obs.xmax, obs.ymin),
            ]
        )
        obs_poly = obs_poly.buffer(inflation)
        obstacle_polys.append(obs_poly)
    if obstacle_polys:
        obstacles_union = unary_union(obstacle_polys)
        free_space = env_poly.difference(obstacles_union)
    else:
        free_space = env_poly
    return free_space


def approximate_bvc_as_polygon(constraints, position, max_radius=20):
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


def compute_safe_region(robot, bvc_constraints, free_space, max_radius=20):
    """
    Compute the safe region for a robot by intersecting its BVC polygon with the free-space.
    """
    bvc_points = approximate_bvc_as_polygon(bvc_constraints, robot.position, max_radius)
    bvc_poly = Polygon(bvc_points)
    safe_region = bvc_poly.intersection(free_space)
    return safe_region


def find_closest_point_in_safe_region(goal, safe_region):
    """
    Given a goal and a safe_region (a Shapely Polygon, MultiPolygon, or GeometryCollection),
    return the point in the region closest to the goal.
    """
    goal_point = Point(goal)
    if safe_region.is_empty:
        return np.array(goal)

    # If safe_region is a GeometryCollection, extract its Polygon components.
    if safe_region.geom_type == "GeometryCollection":
        polys = [geom for geom in safe_region.geoms if geom.geom_type == "Polygon"]
        if polys:
            safe_region = unary_union(polys)
        else:
            return np.array(goal)

    if safe_region.geom_type == "Polygon":
        poly = safe_region
    elif safe_region.geom_type == "MultiPolygon":
        poly = max(safe_region.geoms, key=lambda p: p.area)
    else:
        poly = safe_region  # fallback

    if poly.contains(goal_point):
        return np.array(goal)

    proj_distance = poly.exterior.project(goal_point)
    closest_point = poly.exterior.interpolate(proj_distance)
    return np.array([closest_point.x, closest_point.y])


def simulate_bvc_collision_avoidance(
    robots: list,
    dt=0.1,
    max_steps=1000,
    goal_tolerance=0.1,
    use_right_hand_rule=False,
    obstacles=None,
    environment_size=(40, 40),
):
    # Compute configuration-space free region (obstacles inflated by robot.safety_radius).
    free_space_poly = compute_free_space_polygon(
        environment_size, obstacles, inflation=robots[0].safety_radius
    )
    move_threshold = 1e-3
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
                robot, robots, obstacles=None, use_right_hand_rule=use_right_hand_rule
            )
            safe_region = compute_safe_region(robot, bvc_constraints, free_space_poly)
            # Force target to be inside the free-space: if safe_region is empty, project goal onto free_space.
            if safe_region.is_empty:
                target_point = force_inside_free_space(robot.goal, free_space_poly)
            else:
                target_point = find_closest_point_in_safe_region(robot.goal, safe_region)
                target_point = force_inside_free_space(target_point, free_space_poly)
            # Force movement: if the computed target is too close, force a step toward goal.
            if np.linalg.norm(target_point - robot.position) < move_threshold:
                direction = robot.goal - robot.position
                if np.linalg.norm(direction) > 1e-6:
                    direction = direction / np.linalg.norm(direction)
                    target_point = robot.position + robot.max_speed * dt * direction
                else:
                    target_point = robot.goal
            robot.move_to_point(target_point, dt, free_space=free_space_poly)
    print(
        f"Simulation ended after {max_steps} steps. Not all robots reached their goals."
    )
    return False


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
    environment_size=(40, 40),
):
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
        goal_marker = ax.plot(
            robot.goal[0], robot.goal[1], "x", markersize=10, color=colors[i]
        )[0]
        goal_markers.append(goal_marker)
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

    ax.set_title("BVC Collision Avoidance Animation (Forced Inside Free-space)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.set_aspect("equal")
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, verticalalignment="top")
    all_positions = []
    step = 0
    free_space_poly = compute_free_space_polygon(
        environment_size, obstacles, inflation=sim_robots[0].safety_radius
    )
    while step < max_steps:
        positions = []
        all_reached = True
        for robot in sim_robots:
            if np.linalg.norm(robot.position - robot.goal) > goal_tolerance:
                all_reached = False
                break
        for robot in sim_robots:
            positions.append(robot.position.copy())
            bvc_constraints = compute_buffered_voronoi_cell(
                robot, sim_robots, obstacles=None, use_right_hand_rule=use_right_hand_rule
            )
            safe_region = compute_safe_region(robot, bvc_constraints, free_space_poly)
            if safe_region.is_empty:
                target_point = force_inside_free_space(robot.goal, free_space_poly)
            else:
                target_point = find_closest_point_in_safe_region(robot.goal, safe_region)
                target_point = force_inside_free_space(target_point, free_space_poly)
            if np.linalg.norm(target_point - robot.position) < 1e-3:
                direction = robot.goal - robot.position
                if np.linalg.norm(direction) > 1e-6:
                    direction = direction / np.linalg.norm(direction)
                    target_point = robot.position + robot.max_speed * dt * direction
                else:
                    target_point = robot.goal
            robot.move_to_point(target_point, dt, free_space=free_space_poly)
        all_positions.append((positions, all_reached, step))
        step += 1
        if all_reached:
            break

    def update(frame):
        positions, reached, current_step = all_positions[frame]
        for i, robot in enumerate(sim_robots):
            robot_circles[i].center = robot.trajectory[frame]
            x_data = [pos[0] for pos in robot.trajectory[: frame + 1]]
            y_data = [pos[1] for pos in robot.trajectory[: frame + 1]]
            trajectory_lines[i].set_data(x_data, y_data)
            bvc_constraints = compute_buffered_voronoi_cell(
                robot, sim_robots, obstacles=None, use_right_hand_rule=use_right_hand_rule
            )
            safe_region = compute_safe_region(robot, bvc_constraints, free_space_poly)
            if not safe_region.is_empty:
                if safe_region.geom_type == "Polygon":
                    poly_coords = np.array(safe_region.exterior.coords)
                elif safe_region.geom_type == "MultiPolygon":
                    largest_poly = max(safe_region.geoms, key=lambda p: p.area)
                    poly_coords = np.array(largest_poly.exterior.coords)
                else:
                    poly_coords = np.array([])
                if poly_coords.size > 0:
                    bvc_polygons[i].set_xy(poly_coords)
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
):
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
        environment_size=env_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BVC Simulation with Free-space Intersection for Static Obstacles"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_wj/rect_maps_wj/RectEnv15/agents40/RectEnv_15_40_2.yaml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()
    yaml_config = load_environment(args.config)
    run_yaml_environment(yaml_config, use_right_hand_rule=True, max_steps=1000, dt=0.05)
