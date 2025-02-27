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
        Initialize a robot with single integrator dynamics

        Parameters:
        -----------
        position : numpy.ndarray
            Initial position of the robot [x, y]
        goal : numpy.ndarray
            Goal position of the robot [x, y]
        safety_radius : float
            Safety radius of the robot
        max_speed : float
            Maximum speed of the robot
        id : int
            Robot identifier
        """
        self.position = np.array(position, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.safety_radius = safety_radius
        self.max_speed = max_speed
        self.id = id
        self.trajectory = [self.position.copy()]

    def move_to_point(self, target_point, dt):
        """
        Move the robot toward a target point with limited speed

        Parameters:
        -----------
        target_point : numpy.ndarray
            Target point to move toward
        dt : float
            Time step

        Returns:
        --------
        None
        """
        # Direction to the target
        direction = target_point - self.position
        distance = np.linalg.norm(direction)

        if distance > 0:
            # Normalize direction and scale by speed
            direction = direction / distance
            speed = min(distance / dt, self.max_speed)
            velocity = direction * speed

            # Update position
            self.position = self.position + velocity * dt

        # Store trajectory
        self.trajectory.append(self.position.copy())


class Obstacle:
    def __init__(self, center, width, height):
        """
        Initialize a rectangular obstacle

        Parameters:
        -----------
        center : numpy.ndarray
            Center of the obstacle [x, y]
        width : float
            Width of the obstacle (x-axis)
        height : float
            Height of the obstacle (y-axis)
        """
        self.center = np.array(center, dtype=float)
        self.width = width
        self.height = height

        # Calculate the corners
        self.xmin = self.center[0] - self.width / 2
        self.xmax = self.center[0] + self.width / 2
        self.ymin = self.center[1] - self.height / 2
        self.ymax = self.center[1] + self.height / 2

    def get_closest_point(self, point):
        """
        Get the closest point on the obstacle to the given point

        Parameters:
        -----------
        point : numpy.ndarray
            Point to find closest point to

        Returns:
        --------
        numpy.ndarray
            Closest point on the obstacle
        """
        # Clamp the point to the obstacle boundaries
        closest_x = max(self.xmin, min(point[0], self.xmax))
        closest_y = max(self.ymin, min(point[1], self.ymax))

        return np.array([closest_x, closest_y])

    def distance_to_point(self, point):
        """
        Calculate the distance from the obstacle to the given point

        Parameters:
        -----------
        point : numpy.ndarray
            Point to calculate distance to

        Returns:
        --------
        float
            Distance to the point
        """
        closest_point = self.get_closest_point(point)
        return np.linalg.norm(closest_point - point)

    def is_point_inside(self, point):
        """
        Check if a point is inside the obstacle

        Parameters:
        -----------
        point : numpy.ndarray
            Point to check

        Returns:
        --------
        bool
            True if the point is inside the obstacle, False otherwise
        """
        return self.xmin <= point[0] <= self.xmax and self.ymin <= point[1] <= self.ymax

    def get_constraint_for_point(self, point, safety_radius=0):
        """
        Get the constraint for a point to avoid this obstacle

        Parameters:
        -----------
        point : numpy.ndarray
            Point for which to generate the constraint
        safety_radius : float
            Additional safety radius to consider

        Returns:
        --------
        tuple or None
            Constraint in the form (normal, offset) where normal points away from the obstacle,
            or None if the point is far from the obstacle
        """
        # If the point is inside the obstacle, we need to push it out
        if self.is_point_inside(point):
            closest_point = point.copy()

            # Find the closest edge and push out in that direction
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
            else:  # min_dist == dy_top
                normal = np.array([0.0, 1.0])
                offset = -np.dot(normal, np.array([point[0], self.ymax + safety_radius]))

            return (normal, offset)

        # Find the closest point on the obstacle
        closest_point = self.get_closest_point(point)
        dist = np.linalg.norm(closest_point - point)

        # If the point is close enough to the obstacle, create a constraint
        if dist <= safety_radius:
            # Normal vector pointing away from the obstacle
            if np.linalg.norm(closest_point - point) > 1e-10:
                normal = (point - closest_point) / np.linalg.norm(point - closest_point)
            else:
                # If the point is exactly on the obstacle, use the closest edge normal
                if closest_point[0] == self.xmin:
                    normal = np.array([-1.0, 0.0])
                elif closest_point[0] == self.xmax:
                    normal = np.array([1.0, 0.0])
                elif closest_point[1] == self.ymin:
                    normal = np.array([0.0, -1.0])
                else:  # closest_point[1] == self.ymax
                    normal = np.array([0.0, 1.0])

            # Create a constraint at the safety radius
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
    Compute the Buffered Voronoi Cell for a robot

    Parameters:
    -----------
    robot : Robot
        The robot for which to compute the BVC
    all_robots : list
        List of all robots in the environment
    obstacles : list
        List of obstacles in the environment
    use_right_hand_rule : bool
        Whether to apply the right-hand rule to avoid deadlocks

    Returns:
    --------
    list
        List of inequality constraints defining the BVC
        Each constraint is a tuple (normal, offset) where:
        - normal: normal vector pointing inside the BVC
        - offset: offset from the origin
    """
    constraints = []
    position = robot.position
    safety_radius = robot.safety_radius

    # Direction to goal (used for right-hand rule)
    goal_dir = None
    if use_right_hand_rule:
        goal_vector = robot.goal - position
        if np.linalg.norm(goal_vector) > 1e-6:  # Avoid division by zero
            goal_dir = goal_vector / np.linalg.norm(goal_vector)

    # For each other robot, add a constraint
    for other_robot in all_robots:
        if other_robot.id == robot.id:
            continue

        # Vector from robot to other robot
        p_ij = other_robot.position - position
        p_ij_norm = np.linalg.norm(p_ij)

        # Skip if robots are already in collision (should not happen in normal operation)
        if p_ij_norm < robot.safety_radius + other_robot.safety_radius:
            print(f"Warning: Robots {robot.id} and {other_robot.id} are in collision!")
            continue

        # Normalize the vector
        p_ij_unit = p_ij / p_ij_norm

        # The hyperplane passes through the midpoint plus a safety radius
        midpoint = position + 0.5 * p_ij

        # Apply right-hand rule if enabled and robots are potentially heading for collision
        if use_right_hand_rule and goal_dir is not None:
            # Check if robots are heading towards each other
            # We do this by checking the angle between the goal direction and the other robot
            cos_angle = np.dot(goal_dir, p_ij_unit)

            # If the angle is small (robots heading towards each other)
            if cos_angle > 0.2:  # Threshold can be adjusted
                # Calculate right-hand direction (perpendicular to goal direction, clockwise)
                # In 2D, for vector (x,y), right perpendicular is (y,-x)
                right_dir = np.array([goal_dir[1], -goal_dir[0]])

                # Check if the other robot is more on the right or left side
                side_preference = np.dot(right_dir, p_ij_unit)

                # If the other robot is more on the left, adjust the safety radius
                # to bias movement to the right
                bias_factor = 0.3  # Adjustable
                if side_preference < 0:
                    safety_radius_adjusted = safety_radius * (1 + bias_factor)
                else:
                    safety_radius_adjusted = safety_radius * (1 - bias_factor * 0.5)
            else:
                safety_radius_adjusted = safety_radius
        else:
            safety_radius_adjusted = safety_radius

        offset_point = midpoint - safety_radius_adjusted * p_ij_unit

        # Compute the normal vector (pointing inside the BVC)
        normal = -p_ij_unit

        # Compute the offset (distance from origin to the hyperplane)
        offset = np.dot(normal, offset_point)

        constraints.append((normal, offset))

    # Add constraints for obstacles
    if obstacles:
        for obstacle in obstacles:
            # Get constraint for this obstacle (if needed)
            obstacle_constraint = obstacle.get_constraint_for_point(
                position, safety_radius
            )
            if obstacle_constraint:
                constraints.append(obstacle_constraint)

    return constraints


def is_point_in_bvc(point, constraints):
    """
    Check if a point is inside the BVC

    Parameters:
    -----------
    point : numpy.ndarray
        Point to check
    constraints : list
        List of inequality constraints defining the BVC

    Returns:
    --------
    bool
        True if the point is inside the BVC, False otherwise
    """
    # If there are no constraints, the BVC is unbounded
    if not constraints:
        return True

    for normal, offset in constraints:
        # For a point to be inside a half-space defined by normal·x ≤ offset,
        # we check if normal·point ≤ offset
        if np.dot(normal, point) < offset:
            return False
    return True


def project_point_to_hyperplane(point, hyperplane):
    """
    Project a point onto a hyperplane

    Parameters:
    -----------
    point : numpy.ndarray
        Point to project
    hyperplane : tuple
        Hyperplane defined as (normal, offset)

    Returns:
    --------
    numpy.ndarray
        Projected point
    """
    normal, offset = hyperplane
    normal_unit = normal / np.linalg.norm(normal)

    # Distance from point to hyperplane
    distance = np.dot(normal_unit, point) - offset

    # Project point onto hyperplane
    projection = point - distance * normal_unit

    return projection


def find_closest_point_in_bvc(goal, position, constraints):
    """
    Find the closest point to the goal position within the BVC
    Implementation of Algorithm 1 from the paper

    Parameters:
    -----------
    goal : numpy.ndarray
        Goal position
    position : numpy.ndarray
        Current position of the robot
    constraints : list
        List of inequality constraints defining the BVC

    Returns:
    --------
    numpy.ndarray
        Closest point to the goal within the BVC
    """
    # If there are no constraints, the BVC is unbounded and we can move directly to the goal
    if not constraints:
        return goal.copy()

    # Check if the goal is inside the BVC
    if is_point_in_bvc(goal, constraints):
        return goal.copy()

    # If the goal is outside the BVC, find the closest point on the boundary
    closest_point = None
    min_distance = float("inf")

    # Process each constraint (edge of the BVC)
    for i, (normal_i, offset_i) in enumerate(constraints):
        # Project goal onto the hyperplane
        projection = project_point_to_hyperplane(goal, (normal_i, offset_i))

        # Check if the projection is inside all other constraints
        if is_point_in_bvc(projection, [c for j, c in enumerate(constraints) if j != i]):
            distance = np.linalg.norm(projection - goal)
            if distance < min_distance:
                min_distance = distance
                closest_point = projection

    # If no valid projection found (we need to find a vertex)
    if closest_point is None:
        # For each pair of constraints, find their intersection (vertex)
        for i in range(len(constraints)):
            for j in range(i + 1, len(constraints)):
                normal_i, offset_i = constraints[i]
                normal_j, offset_j = constraints[j]

                # Find the intersection of two hyperplanes
                # Solve the system: normal_i⋅p = offset_i, normal_j⋅p = offset_j
                # In 2D, we can solve this directly
                A = np.vstack([normal_i, normal_j])
                b = np.array([offset_i, offset_j])

                try:
                    vertex = np.linalg.solve(A, b)

                    # Check if the vertex is inside all other constraints
                    if is_point_in_bvc(
                        vertex,
                        [c for k, c in enumerate(constraints) if k != i and k != j],
                    ):
                        distance = np.linalg.norm(vertex - goal)
                        if distance < min_distance:
                            min_distance = distance
                            closest_point = vertex
                except np.linalg.LinAlgError:
                    # Hyperplanes are parallel or nearly parallel
                    continue

    # If still no closest point found (very rare), move towards the goal but with a small step
    if closest_point is None:
        direction = goal - position
        distance = np.linalg.norm(direction)
        if distance > 1e-6:
            direction = direction / distance
            closest_point = position + 0.1 * direction  # Small step towards goal
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
    Simulate the BVC collision avoidance algorithm

    Parameters:
    -----------
    robots : list
        List of robots
    dt : float
        Time step
    max_steps : int
        Maximum number of simulation steps
    goal_tolerance : float
        Distance tolerance to consider a goal reached
    use_right_hand_rule : bool
        Whether to apply the right-hand rule to avoid deadlocks
    obstacles : list
        List of obstacles in the environment

    Returns:
    --------
    bool
        True if all robots reached their goals, False otherwise
    """
    # Main simulation loop
    for step in range(max_steps):
        # Check if all robots reached their goals
        all_reached = True
        for robot in robots:
            distance_to_goal = np.linalg.norm(robot.position - robot.goal)
            if distance_to_goal > goal_tolerance:
                all_reached = False
                break

        if all_reached:
            print(f"All robots reached their goals in {step} steps!")
            return True

        # Update each robot
        for robot in robots:
            # Compute BVC for the robot with right-hand rule if enabled
            bvc_constraints = compute_buffered_voronoi_cell(
                robot, robots, obstacles, use_right_hand_rule
            )

            # Find closest point in BVC to the goal
            target_point = find_closest_point_in_bvc(
                robot.goal, robot.position, bvc_constraints
            )

            # Move toward the target point
            robot.move_to_point(target_point, dt)

    print(
        f"Simulation ended after {max_steps} steps. Not all robots reached their goals."
    )
    return False


def visualize_simulation(robots, figure_size=(10, 10), boundary=None, obstacles=None):
    """
    Visualize the simulation results

    Parameters:
    -----------
    robots : list
        List of robots
    figure_size : tuple
        Size of the figure
    boundary : tuple
        Boundary of the environment (xmin, xmax, ymin, ymax)
    obstacles : list
        List of obstacles in the environment

    Returns:
    --------
    None
    """
    plt.figure(figsize=figure_size)

    # Set boundary if provided
    if boundary:
        xmin, xmax, ymin, ymax = boundary
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

    # Draw obstacles
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

    # Plot trajectories
    for robot in robots:
        trajectory = np.array(robot.trajectory)
        plt.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            "-",
            linewidth=2,
            label=f"Robot {robot.id}",
        )

        # Mark start position
        plt.plot(trajectory[0, 0], trajectory[0, 1], "o", markersize=10)

        # Mark goal position
        plt.plot(robot.goal[0], robot.goal[1], "x", markersize=10)

        # Draw final position with safety radius
        circle = plt.Circle(robot.position, robot.safety_radius, fill=True, alpha=0.3)
        plt.gca().add_patch(circle)

    plt.grid(True)
    plt.legend()
    plt.title("Buffered Voronoi Cell Collision Avoidance Simulation")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.show()


def approximate_bvc_as_polygon(constraints, position, max_radius=10):
    """
    Approximate a BVC as a polygon for visualization

    Parameters:
    -----------
    constraints : list
        List of inequality constraints defining the BVC
    position : numpy.ndarray
        Position of the robot
    max_radius : float
        Maximum radius from the position to consider if no constraint in a direction

    Returns:
    --------
    numpy.ndarray
        Polygon points approximating the BVC
    """
    if not constraints:
        # Return a circle if no constraints
        angles = np.linspace(0, 2 * np.pi, 20)
        circle_points = position + max_radius * np.column_stack(
            [np.cos(angles), np.sin(angles)]
        )
        return circle_points

    # Generate rays in different directions from the position
    num_angles = 36  # Number of directions to check
    angles = np.linspace(0, 2 * np.pi, num_angles)
    directions = np.column_stack([np.cos(angles), np.sin(angles)])

    polygon_points = []
    for direction in directions:
        # Find the intersection with the closest constraint
        min_distance = max_radius
        for normal, offset in constraints:
            # Check if ray intersects with the constraint
            normal_dot_dir = np.dot(normal, direction)
            if abs(normal_dot_dir) > 1e-10:  # Avoid division by zero
                # Distance to intersection
                distance = (offset - np.dot(normal, position)) / normal_dot_dir
                if 0 < distance < min_distance:
                    min_distance = distance

        # Add the point at the found distance
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
    Animate the BVC collision avoidance algorithm

    Parameters:
    -----------
    robots : list
        List of robots
    dt : float
        Time step
    max_steps : int
        Maximum number of simulation steps
    goal_tolerance : float
        Distance tolerance to consider a goal reached
    figure_size : tuple
        Size of the figure
    boundary : tuple
        Boundary of the environment (xmin, xmax, ymin, ymax)
    interval : int
        Interval between animation frames in milliseconds
    save_animation : bool
        Whether to save the animation as a file
    use_right_hand_rule : bool
        Whether to apply the right-hand rule to avoid deadlocks
    obstacles : list
        List of obstacles in the environment

    Returns:
    --------
    None
    """
    # Make a copy of the robots to avoid modifying the original
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

    # Setup the figure
    fig, ax = plt.subplots(figsize=figure_size)
    if boundary:
        xmin, xmax, ymin, ymax = boundary
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    # Initialize plot elements
    robot_circles = []
    goal_markers = []
    trajectory_lines = []
    bvc_polygons = []

    # Colors for each robot
    colors = plt.cm.tab10(np.linspace(0, 1, len(sim_robots)))

    # Initialize plots for each robot
    for i, robot in enumerate(sim_robots):
        # Robot position with safety radius
        circle = plt.Circle(
            robot.position, robot.safety_radius, fill=True, alpha=0.5, color=colors[i]
        )
        robot_circles.append(ax.add_patch(circle))

        # Robot goal
        goal = ax.plot(robot.goal[0], robot.goal[1], "x", markersize=10, color=colors[i])[
            0
        ]
        goal_markers.append(goal)

        # Robot trajectory, initially empty
        (trajectory,) = ax.plot(
            [], [], "-", linewidth=1.5, color=colors[i], label=f"Robot {robot.id}"
        )
        trajectory_lines.append(trajectory)

        # BVC polygon, initially empty
        polygon = patches.Polygon(
            np.zeros((1, 2)), closed=True, fill=False, edgecolor=colors[i], alpha=0.3
        )
        bvc_polygons.append(ax.add_patch(polygon))

    # Add obstacles to the animation
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

    # Add title, legend, and grid
    ax.set_title("Buffered Voronoi Cell Collision Avoidance Animation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.legend(loc="upper left")
    ax.set_aspect("equal")

    # Text for simulation info
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, verticalalignment="top")

    # List to store all robot positions for each step
    all_positions = []

    # Initialize simulation parameters
    step = 0
    all_reached = False

    # Precompute the simulation to avoid computation during animation
    while step < max_steps and not all_reached:
        positions = []

        # Check if all robots reached their goals
        all_reached = True
        for robot in sim_robots:
            distance_to_goal = np.linalg.norm(robot.position - robot.goal)
            if distance_to_goal > goal_tolerance:
                all_reached = False
                break

        # Update each robot
        for robot in sim_robots:
            positions.append(robot.position.copy())

            # Compute BVC for the robot
            bvc_constraints = compute_buffered_voronoi_cell(
                robot, sim_robots, obstacles, use_right_hand_rule
            )

            # Find closest point in BVC to the goal
            target_point = find_closest_point_in_bvc(
                robot.goal, robot.position, bvc_constraints
            )

            # Move toward the target point
            robot.move_to_point(target_point, dt)

        all_positions.append((positions, all_reached, step))
        step += 1

    # Animation update function
    def update(frame):
        positions, reached, current_step = all_positions[frame]

        # Update robot positions and trajectories
        for i, robot in enumerate(sim_robots):
            # Update robot position
            robot_circles[i].center = robot.trajectory[frame]

            # Update trajectory
            x_data = [pos[0] for pos in robot.trajectory[: frame + 1]]
            y_data = [pos[1] for pos in robot.trajectory[: frame + 1]]
            trajectory_lines[i].set_data(x_data, y_data)

            # Compute and update BVC
            bvc_constraints = compute_buffered_voronoi_cell(
                robot, sim_robots, obstacles, use_right_hand_rule
            )
            polygon_points = approximate_bvc_as_polygon(bvc_constraints, robot.position)
            bvc_polygons[i].set_xy(polygon_points)

        # Update simulation info
        status = "COMPLETE" if reached else "IN PROGRESS"
        info_text.set_text(f"Step: {current_step} | Status: {status}")

        return robot_circles + trajectory_lines + bvc_polygons + [info_text]

    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(all_positions), interval=interval, blit=True
    )

    # Save animation if requested
    if save_animation:
        anim.save("bvc_collision_avoidance.mp4", writer="ffmpeg", fps=30)

    plt.tight_layout()
    plt.show()


def load_environment(yaml_file):
    """
    Load environment configuration from YAML file

    Parameters:
    -----------
    yaml_file : str
        Path to the YAML file

    Returns:
    --------
    dict
        Environment configuration
    """
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_environment_from_yaml(yaml_file, robot_radius=0.2, max_speed=0.8):
    """
    Create robots and obstacles from a YAML configuration file

    Parameters:
    -----------
    yaml_file : str
        Path to the YAML file
    robot_radius : float
        Radius of the robots
    max_speed : float
        Maximum speed of the robots

    Returns:
    --------
    tuple
        (robots, obstacles, environment_size)
    """
    # Load configuration
    config = None
    if isinstance(yaml_file, str):
        with open(yaml_file, "r") as file:
            config = yaml.safe_load(file)
    else:
        # Assume it's already a dict
        config = yaml_file

    # Create robots
    robots = []
    for i in range(config["agentNum"]):
        start_point = config["startPoints"][i]
        goal_point = config["goalPoints"][i]
        robot = Robot(start_point, goal_point, robot_radius, max_speed=max_speed, id=i)
        robots.append(robot)

    # Create obstacles
    obstacles = []
    for obs_config in config["obstacles"]:
        center = obs_config["center"]
        width = obs_config["width"]
        height = obs_config["height"]
        obstacle = Obstacle(center, width, height)
        obstacles.append(obstacle)

    # Default environment size is 40x40
    environment_size = (40, 40)

    return robots, obstacles, environment_size


def run_yaml_environment(yaml_config, use_right_hand_rule=True, max_steps=1000, dt=0.05):
    """
    Run simulation with environment loaded from YAML

    Parameters:
    -----------
    yaml_config : str or dict
        YAML file path or already loaded configuration
    use_right_hand_rule : bool
        Whether to use the right-hand rule to avoid deadlocks
    max_steps : int
        Maximum number of simulation steps
    dt : float
        Time step for simulation

    Returns:
    --------
    None
    """
    # Create environment
    robots, obstacles, env_size = create_environment_from_yaml(yaml_config)

    # Set boundary for animation
    boundary = (0, env_size[0], 0, env_size[1])

    # Animate the simulation
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
    # yaml_config = load_environment("RectEnv_10_20_0.yaml")
    parser = argparse.ArgumentParser(description="ORCA Simulation with configurable path")
    parser.add_argument(
        "--config",
        type=str,
        # default="/mnt/Topics/Learning/CBF/map_generator/circle_maps_wj/CircleEnv15/agents100/CircleEnv_15_100_11.yaml",
        # default="/mnt/Topics/Learning/CBF/map_generator/Free_maps_wj/agents120/Free_0_120_5.yaml",
        # default="/mnt/Topics/Learning/CBF/map_generator/Free_maps_wj/mapf_env_100agents.yaml",
        default="benchmark_wj/rect_maps_wj/RectEnv15/agents20/RectEnv_15_20_2.yaml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()
    yaml_config = load_environment(args.config)

    run_yaml_environment(yaml_config, use_right_hand_rule=True, max_steps=2000)

    # Load and run the YAML environment
    # run_yaml_environment({
    #     'agentNum': 10,
    #     'startPoints': [
    #         [25.895226782387695, 21.179512215464356],
    #         [29.48605861497971, 7.323474008430857],
    #         [21.66564602120946, 34.13826436590728],
    #         [11.112549986462614, 18.55653080489052],
    #         [30.064954880169836, 10.260805713912983],
    #         [8.993982885755122, 10.74428553661661],
    #         [6.7402181593370205, 7.6301862358265415],
    #         [3.420866309810507, 28.887607292676755],
    #         [15.374001487087032, 37.98675952110221],
    #         [8.173926275344227, 20.713459924856565]
    #     ],
    #     'goalPoints': [
    #         [34.541093266567145, 18.081240300924925],
    #         [22.910800533073886, 4.25313503607252],
    #         [14.32702325352887, 28.53252390394023],
    #         [28.4596552760862, 36.97447803014795],
    #         [34.158424000943405, 29.697384817527297],
    #         [3.988689935738609, 13.06456292149025],
    #         [13.584146630154974, 11.050850490253696],
    #         [11.209948635567574, 9.513065374778607],
    #         [34.35641934574286, 37.48081216261031],
    #         [24.94213107466699, 5.242372578365997]
    #     ],
    #     'obstacles': [
    #         {'center': [32.42075276318654, 8.456272210888233], 'height': 4.668712638667812,
    #          'width': 1.5562375462226044},
    #         {'center': [33.976990309409146, 6.900034664665631], 'height': 1.5562375462226044,
    #          'width': 1.5562375462226044},
    #         {'center': [14.245653438262865, 13.949538205886494], 'height': 3.3179862040621337,
    #          'width': 3.3179862040621337},
    #         {'center': [3.727491248001829, 25.560846637638985], 'height': 1.5660219282712333,
    #          'width': 3.132043856542462},
    #         {'center': [2.161469319730598, 23.99482470936775], 'height': 1.5660219282712333,
    #          'width': 3.132043856542463},
    #         {'center': [31.204171378076214, 32.17743557578943], 'height': 3.1468907690103123,
    #          'width': 1.5734453845051561},
    #         {'center': [32.777616762581374, 33.75088096029459], 'height': 3.1468907690103123,
    #          'width': 1.5734453845051597},
    #         {'center': [14.814888464827899, 4.731148456056243], 'height': 4.937512880640112,
    #          'width': 1.6458376268800379},
    #         {'center': [13.16905083794786, 6.3769860829362806], 'height': 1.6458376268800379,
    #          'width': 1.6458376268800379},
    #         {'center': [34.81309853448139, 23.25425314467011], 'height': 3.0896199239815303,
    #          'width': 1.5448099619907651},
    #         {'center': [33.268288572490626, 21.709443182679347], 'height': 3.0896199239815303,
    #          'width': 1.5448099619907651},
    #         {'center': [6.559523956652588, 0.9202136576966561], 'height': 1.5941365310327444,
    #          'width': 1.5941365310327438},
    #         {'center': [4.965387425619843, 2.5143501887294004], 'height': 1.5941365310327447,
    #          'width': 4.782409593098233},
    #         {'center': [22.71789032641157, 26.92229780849816], 'height': 1.6203230386747975,
    #          'width': 1.6203230386747975},
    #         {'center': [21.097567287736773, 28.54262084717296], 'height': 1.6203230386747975,
    #          'width': 4.860969116024393},
    #         {'center': [33.24590104703481, 13.888562811728967], 'height': 1.6073748137175947,
    #          'width': 6.429499254870379},
    #         {'center': [6.181058717362433, 15.365133353938166], 'height': 1.5584501856818846,
    #          'width': 3.116900371363762},
    #         {'center': [7.739508903044314, 16.923583539620047], 'height': 1.5584501856818775,
    #          'width': 3.116900371363762},
    #         {'center': [15.035506282658485, 32.99817531883738], 'height': 1.5012364710384531,
    #          'width': 3.0024729420769063},
    #         {'center': [13.534269811620032, 31.49693884779893], 'height': 1.5012364710384531,
    #          'width': 3.0024729420769063},
    #         {'center': [23.65946465406624, 12.866508568631165], 'height': 1.525845398877351,
    #          'width': 4.577536196632046},
    #         {'center': [23.65946465406624, 11.340663169753816], 'height': 1.5258453988773475,
    #          'width': 1.525845398877351},
    #         {'center': [27.173340288161484, 26.29368701721762], 'height': 3.2908486727541515,
    #          'width': 1.6454243363770757},
    #         {'center': [28.81876462453856, 24.648262680840546], 'height': 3.2908486727541515,
    #          'width': 1.6454243363770757},
    #         {'center': [38.20457019787727, 33.26109874085263], 'height': 3.3190128143657063,
    #          'width': 3.3190128143657063},
    #         {'center': [8.921273356439945, 36.02973621303944], 'height': 6.148476597523697,
    #          'width': 1.5371191493809242},
    #         {'center': [18.3599819993779, 23.051201201971708], 'height': 3.1933049867145584,
    #          'width': 3.1933049867145584}
    #     ]
    # }, use_right_hand_rule=True, max_steps=2000)
