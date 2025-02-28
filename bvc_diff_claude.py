"""
Buffered Voronoi Cell Collision Avoidance Simulation

This module implements collision avoidance for multiple robots using the Buffered Voronoi Cell approach.
It supports different types of environments with both rectangular and circular obstacles.

Main features:
- BVC-based collision avoidance
- Support for rectangular and circular obstacles
- Differential drive robot kinematics
- Recovery behavior for deadlock situations
- Visualization and animation capabilities
"""

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import patches, animation
import yaml
import argparse
import time
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import nearest_points

# Constants
GLOBAL_SCALE = 2.5


# -----------------------------------------------------------------------------
# Robot Implementation
# -----------------------------------------------------------------------------
class Robot:
    """
    Differential drive robot with BVC collision avoidance capabilities.
    """

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
        Initialize a differential drive robot.

        Parameters:
        -----------
        position : numpy.ndarray
            Initial position [x, y]
        goal : numpy.ndarray
            Goal position [x, y]
        safety_radius : float
            Safety radius for collision avoidance
        max_speed : float
            Maximum linear speed
        max_angular : float
            Maximum angular speed
        initial_theta : float
            Initial orientation (radians)
        id : int
            Robot identifier
        """
        # Basic properties
        self.position = np.array(position, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.safety_radius = safety_radius
        self.max_speed = max_speed
        self.max_angular = max_angular
        self.theta = initial_theta
        self.id = id

        # Trajectory recording
        self.trajectory = [self.position.copy()]
        self.theta_trajectory = [self.theta]

        # Collision and recovery properties
        self.stuck_counter = 0
        self.last_position = self.position.copy()
        self.in_collision = False
        self.recovery_mode = False
        self.recovery_direction = None
        self.recovery_steps_left = 0
        self.prev_position = self.position.copy()

    def move_to_point(self, target_point, dt, obstacles=None, boundary=None):
        """
        Move the robot toward the target point using differential drive kinematics.
        The linear speed is modulated by the cosine of the heading error.
        Includes checks to prevent obstacle/boundary crossing.
        """
        # Save previous position for crossing detection
        self.prev_position = self.position.copy()

        # Detect if robot is stuck
        position_diff = np.linalg.norm(self.position - self.last_position)
        self.last_position = self.position.copy()

        if position_diff < 0.01 * dt:
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)

        # Recovery mode management
        if self.recovery_mode and position_diff > 0.05:
            self.recovery_mode = False

        # Determine desired heading
        if self.recovery_mode and self.recovery_steps_left > 0:
            desired_angle = self.recovery_direction
            self.recovery_steps_left -= 1
        else:
            desired_angle = np.arctan2(
                target_point[1] - self.position[1], target_point[0] - self.position[0]
            )

        # Calculate control inputs
        angle_error = desired_angle - self.theta
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

        # Angular control
        Kp_ang = 4.0  # 10.0
        omega = np.clip(Kp_ang * angle_error, -self.max_angular, self.max_angular)

        # Linear speed control
        distance = np.linalg.norm(target_point - self.position)
        v = self.max_speed * max(0, np.cos(angle_error))
        v = min(v, distance / dt)

        # Speed adjustments
        if self.recovery_mode:
            v *= 0.5

        # Boundary speed reduction
        if boundary:
            xmin, xmax, ymin, ymax = boundary
            distance_to_boundary = min(
                self.position[0] - xmin,
                xmax - self.position[0],
                self.position[1] - ymin,
                ymax - self.position[1],
            )
            boundary_factor = min(1.0, distance_to_boundary / (2.0 * self.safety_radius))
            v *= boundary_factor

        # Calculate new position and orientation
        new_position = self.position.copy()
        new_position[0] += v * np.cos(self.theta) * dt
        new_position[1] += v * np.sin(self.theta) * dt
        new_theta = self.theta + omega * dt
        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi

        # Safety checks
        would_cross_obstacle = False
        if obstacles:
            for obstacle in obstacles:
                if obstacle.would_movement_cross(
                    self.position, new_position, self.safety_radius
                ):
                    would_cross_obstacle = True
                    break

        would_cross_boundary = False
        if boundary:
            xmin, xmax, ymin, ymax = boundary
            margin = self.safety_radius * 0.5
            if (
                new_position[0] < xmin + margin
                or new_position[0] > xmax - margin
                or new_position[1] < ymin + margin
                or new_position[1] > ymax - margin
            ):
                would_cross_boundary = True

        # Update position if safe
        if not would_cross_obstacle and not would_cross_boundary:
            self.position = new_position
            self.theta = new_theta
        else:
            self.stuck_counter += 2
            v = 0

        # Record trajectory
        self.trajectory.append(self.position.copy())
        self.theta_trajectory.append(self.theta)

    def initiate_recovery(self, obstacles=None, all_robots=None, boundary=None):
        """
        Initiate recovery behavior when robot is stuck or in collision.
        Robots that have reached their goals should not enter recovery mode.
        """
        # Check if robot has already reached its goal.
        distance_to_goal = np.linalg.norm(self.position - self.goal)
        if distance_to_goal < self.safety_radius:
            self.recovery_mode = False
            self.recovery_steps_left = 0
            return

        # Attempt to compute the current BVC to see if we are stuck at a vertex.
        try:
            bvc_poly = compute_BVC_polygon(
                self, all_robots, obstacles, boundary, use_right_hand_rule=False
            )
        except Exception:
            bvc_poly = None

        if bvc_poly and not bvc_poly.is_empty:
            # If bvc_poly is a MultiPolygon, select the largest polygon.
            if bvc_poly.geom_type == "MultiPolygon":
                bvc_poly = max(bvc_poly.geoms, key=lambda p: p.area)
            # Extract vertices from the BVC polygon.
            vertices = np.array(bvc_poly.exterior.coords)
            # Compute distances from the robot's current position to each vertex.
            distances = np.linalg.norm(vertices - self.position, axis=1)
            # If the robot is very close to any vertex, consider it "stuck" at that vertex.
            if np.any(distances < 0.05):  # 0.05 is a tunable threshold.
                idx = np.argmin(distances)
                num_vertices = len(vertices)
                # Determine the adjacent vertices.
                prev_vertex = vertices[idx - 1]
                next_vertex = vertices[(idx + 1) % num_vertices]
                # Compute a recovery (detour) direction along the edge.
                edge_dir = next_vertex - prev_vertex
                norm_edge = np.linalg.norm(edge_dir)
                if norm_edge > 1e-6:
                    edge_dir /= norm_edge
                else:
                    edge_dir = np.array([1.0, 0.0])
                # Use the sign determined by comparing with the goal direction.
                goal_dir = self.goal - self.position
                if np.linalg.norm(goal_dir) > 1e-6:
                    goal_dir /= np.linalg.norm(goal_dir)
                else:
                    goal_dir = np.array([1.0, 0.0])
                # Choose the direction that deviates from the line toward the goal.
                if np.cross(edge_dir, goal_dir) > 0:
                    recovery_dir = edge_dir
                else:
                    recovery_dir = -edge_dir
                self.recovery_direction = np.arctan2(recovery_dir[1], recovery_dir[0])
                self.recovery_mode = True
                self.recovery_steps_left = 20
                return

        # Fallback to potential-field recovery if not stuck at a vertex.
        self.recovery_mode = True
        self.recovery_steps_left = 20
        recovery_vector = np.zeros(2)

        # Add repulsion from obstacles.
        if obstacles:
            for obstacle in obstacles:
                closest = obstacle.get_closest_point(self.position)
                vec = self.position - closest
                dist = np.linalg.norm(vec)
                if dist < 1e-6:
                    vec = np.array([np.random.rand() - 0.5, np.random.rand() - 0.5])
                else:
                    vec = vec / dist
                strength = min(1.0, 0.5 / max(0.1, dist))
                recovery_vector += vec * strength

        # Add repulsion from other robots.
        if all_robots:
            for other in all_robots:
                if other.id == self.id:
                    continue
                vec = self.position - other.position
                dist = np.linalg.norm(vec)
                if dist < 1e-6:
                    vec = np.array([np.random.rand() - 0.5, np.random.rand() - 0.5])
                else:
                    vec = vec / dist
                strength = min(1.0, 0.5 / max(0.1, dist))
                recovery_vector += vec * strength

        # Add repulsion from boundaries.
        if boundary:
            xmin, xmax, ymin, ymax = boundary
            margin = self.safety_radius * 2.0
            # Left boundary
            dist_left = self.position[0] - xmin
            if dist_left < margin:
                strength = min(1.0, 1.0 / max(0.1, dist_left / margin))
                recovery_vector += np.array([strength, 0.0])
            # Right boundary
            dist_right = xmax - self.position[0]
            if dist_right < margin:
                strength = min(1.0, 1.0 / max(0.1, dist_right / margin))
                recovery_vector += np.array([-strength, 0.0])
            # Bottom boundary
            dist_bottom = self.position[1] - ymin
            if dist_bottom < margin:
                strength = min(1.0, 1.0 / max(0.1, dist_bottom / margin))
                recovery_vector += np.array([0.0, strength])
            # Top boundary
            dist_top = ymax - self.position[1]
            if dist_top < margin:
                strength = min(1.0, 1.0 / max(0.1, dist_top / margin))
                recovery_vector += np.array([0.0, -strength])

        if np.linalg.norm(recovery_vector) < 1e-6:
            recovery_vector = np.array([np.random.rand() - 0.5, np.random.rand() - 0.5])
        recovery_vector = recovery_vector / max(1e-6, np.linalg.norm(recovery_vector))
        self.recovery_direction = np.arctan2(recovery_vector[1], recovery_vector[0])


# -----------------------------------------------------------------------------
# Obstacle Implementation
# -----------------------------------------------------------------------------
class BaseObstacle:
    """Base class for all obstacles"""

    def get_closest_point(self, point):
        """Return the closest point on the obstacle's boundary to the given point."""
        raise NotImplementedError("Subclasses must implement this method")

    def distance_to_point(self, point):
        """Calculate distance from a point to the obstacle."""
        closest_point = self.get_closest_point(point)
        return np.linalg.norm(closest_point - point)

    def is_point_inside(self, point):
        """Check if a point is inside the obstacle."""
        return self.polygon.contains(Point(point))

    def would_movement_cross(self, start_point, end_point, safety_radius=0):
        """Check if a movement from start_point to end_point would cross this obstacle."""
        line = LineString([tuple(start_point), tuple(end_point)])
        buffered_polygon = self.polygon.buffer(safety_radius)
        return buffered_polygon.intersects(line)

    def get_constraint_for_point(self, point, safety_radius=0):
        """Generate a half-plane constraint for collision avoidance."""
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


class Obstacle(BaseObstacle):
    """Rectangular obstacle implementation."""

    def __init__(self, center, width, height):
        """Initialize a rectangular obstacle."""
        self.center = np.array(center, dtype=float)
        self.width = width
        self.height = height

        self.xmin = self.center[0] - self.width / 2
        self.xmax = self.center[0] + self.width / 2
        self.ymin = self.center[1] - self.height / 2
        self.ymax = self.center[1] + self.height / 2

        # Create polygon representation
        self.polygon = Polygon(
            [
                (self.xmin, self.ymin),
                (self.xmin, self.ymax),
                (self.xmax, self.ymax),
                (self.xmax, self.ymin),
            ]
        )

        # Cache vertices for faster access
        self.vertices = [
            (self.xmin, self.ymin),
            (self.xmin, self.ymax),
            (self.xmax, self.ymax),
            (self.xmax, self.ymin),
        ]

    def get_closest_point(self, point):
        """Return the closest point on the obstacle's boundary to the given point."""
        p = Point(point)
        proj_dist = self.polygon.exterior.project(p)
        closest = self.polygon.exterior.interpolate(proj_dist)
        return np.array([closest.x, closest.y])

    def get_constraints_for_point(self, point, safety_radius=0):
        """Generate multiple constraints for better corner handling."""
        p = np.array(point, dtype=float)
        p_shapely = Point(p)

        # Skip if point is far from obstacle
        if self.polygon.distance(p_shapely) > safety_radius * 2:
            return []

        constraints = []
        is_inside = self.polygon.contains(p_shapely)
        closest = self.get_closest_point(p)

        # Check for proximity to corners
        corners = np.array(self.vertices)
        corner_distances = np.linalg.norm(corners - p, axis=1)
        near_corners = corner_distances < safety_radius * 1.5

        # Add basic constraint
        if is_inside or self.polygon.distance(p_shapely) < safety_radius:
            diff = p - closest
            norm = np.linalg.norm(diff)

            if norm < 1e-6:
                normal = np.array([1.0, 0.0])
            else:
                normal = diff / norm

            constraint_point = closest + safety_radius * normal
            offset = np.dot(normal, constraint_point)
            constraints.append((normal, offset))

            # Add additional constraints for corners
            if np.any(near_corners):
                for i, is_near in enumerate(near_corners):
                    if is_near:
                        corner = corners[i]
                        prev_idx = (i - 1) % 4
                        next_idx = (i + 1) % 4
                        prev_corner = corners[prev_idx]
                        next_corner = corners[next_idx]

                        # Create edge normals
                        edge1 = corner - prev_corner
                        edge2 = next_corner - corner

                        for edge in [edge1, edge2]:
                            edge_normal = np.array([-edge[1], edge[0]])
                            edge_normal = edge_normal / np.linalg.norm(edge_normal)

                            # Ensure outward direction
                            center_to_point = p - self.center
                            if np.dot(edge_normal, center_to_point) < 0:
                                edge_normal = -edge_normal

                            constraint_point = corner + safety_radius * edge_normal
                            offset = np.dot(edge_normal, constraint_point)
                            constraints.append((edge_normal, offset))

        return constraints


class CircleObstacle(BaseObstacle):
    """Circular obstacle implementation."""

    def __init__(self, center, radius):
        """Initialize a circular obstacle."""
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.polygon = Point(self.center).buffer(self.radius, resolution=64)

    def get_closest_point(self, point):
        """Return the closest point on the obstacle's boundary to the given point."""
        p = Point(point)
        proj_dist = self.polygon.exterior.project(p)
        closest = self.polygon.exterior.interpolate(proj_dist)
        return np.array([closest.x, closest.y])


# -----------------------------------------------------------------------------
# BVC Core Functions
# -----------------------------------------------------------------------------
def create_boundary_constraints(robot, boundary, safety_margin=0.0):
    """Create constraints representing the environment boundaries."""
    if boundary is None:
        return []

    xmin, xmax, ymin, ymax = boundary
    safety_radius = robot.safety_radius + safety_margin

    constraints = []

    # Left boundary: normal points right (1, 0)
    normal = np.array([1.0, 0.0])
    offset = normal[0] * (xmin + safety_radius)
    constraints.append((normal, offset))

    # Right boundary: normal points left (-1, 0)
    normal = np.array([-1.0, 0.0])
    offset = normal[0] * (xmax - safety_radius)
    constraints.append((normal, offset))

    # Bottom boundary: normal points up (0, 1)
    normal = np.array([0.0, 1.0])
    offset = normal[1] * (ymin + safety_radius)
    constraints.append((normal, offset))

    # Top boundary: normal points down (0, -1)
    normal = np.array([0.0, -1.0])
    offset = normal[1] * (ymax - safety_radius)
    constraints.append((normal, offset))

    return constraints


def compute_buffered_voronoi_cell(
    robot: Robot,
    all_robots: list,
    obstacles: list[Obstacle] = None,
    boundary=None,
    use_right_hand_rule=False,
    sensing_radius=10.0,
):
    """
    Compute the Buffered Voronoi Cell (BVC) for a robot.
    Returns a list of constraints that define the BVC.
    """
    constraints = []
    scale_factor = 1.5
    safety_radius = robot.safety_radius * scale_factor
    position = robot.position
    goal_dir = None

    # Add boundary constraints
    if boundary:
        boundary_safety_margin = 1.0
        boundary_constraints = create_boundary_constraints(
            robot, boundary, boundary_safety_margin
        )
        constraints.extend(boundary_constraints)

    # Calculate goal direction for right-hand rule
    if use_right_hand_rule:
        goal_vector = robot.goal - position
        if np.linalg.norm(goal_vector) > 1e-6:
            goal_dir = goal_vector / np.linalg.norm(goal_vector)

    # Find nearby robots for efficiency
    nearby_robots = []
    for other_robot in all_robots:
        if other_robot.id == robot.id:
            continue
        if np.linalg.norm(other_robot.position - position) <= sensing_radius:
            nearby_robots.append(other_robot)

    # Add constraints from neighboring robots
    for other_robot in nearby_robots:
        if other_robot.id == robot.id:
            continue

        p_ij = other_robot.position - position
        p_ij_norm = np.linalg.norm(p_ij)

        # If the robots are too close, mark collision but do not immediately exit.
        if p_ij_norm < (robot.safety_radius + other_robot.safety_radius):
            robot.in_collision = True
            # Optionally, you could adjust p_ij_norm here or log a warning
            # but continue computing the constraint.

        # Compute unit vector and midpoint as before.
        p_ij_unit = p_ij / p_ij_norm if p_ij_norm > 1e-6 else np.array([1.0, 0.0])
        midpoint = position + 0.5 * p_ij

        # Apply right-hand rule if enabled.
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

        # Create constraint using adjusted safety radius.
        offset_point = midpoint - safety_radius_adjusted * p_ij_unit
        normal = -p_ij_unit
        offset = np.dot(normal, offset_point)
        constraints.append((normal, offset))

    # Add obstacle constraints
    if obstacles:
        for obstacle in obstacles:
            # Skip distant obstacles
            obstacle_size = (
                getattr(obstacle, "radius", 0)
                or max(getattr(obstacle, "width", 0), getattr(obstacle, "height", 0)) / 2
            )

            if (
                np.linalg.norm(obstacle.center - position)
                > sensing_radius + obstacle_size
            ):
                continue

            # Get constraint based on obstacle type
            if hasattr(obstacle, "get_constraints_for_point"):
                obs_constraints = obstacle.get_constraints_for_point(
                    position, safety_radius
                )
                if obs_constraints:
                    constraints.extend(obs_constraints)
            else:
                obs_constraint = obstacle.get_constraint_for_point(
                    position, safety_radius
                )
                if obs_constraint:
                    constraints.append(obs_constraint)

    return constraints


def compute_BVC_polygon(
    robot: Robot,
    all_robots: list[Robot],
    obstacles: list[Obstacle] = None,
    boundary=None,
    use_right_hand_rule=False,
):
    """
    Compute the BVC as a shapely Polygon.
    First, compute the approximate polygon from neighbor constraints.
    Then subtract the obstacles from it.
    """
    constraints = compute_buffered_voronoi_cell(
        robot, all_robots, obstacles, boundary, use_right_hand_rule
    )

    if not constraints:
        return Polygon()  # Empty polygon

    # Create initial polygon from constraints
    poly_points = approximate_bvc_as_polygon(constraints, robot.position)
    bvc_poly = Polygon(poly_points)

    # Clip to boundary
    if boundary:
        xmin, xmax, ymin, ymax = boundary
        boundary_poly = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
        bvc_poly = bvc_poly.intersection(boundary_poly)

    # Subtract obstacles
    if obstacles:
        for obs in obstacles:
            bvc_poly = bvc_poly.difference(obs.polygon)

    return bvc_poly


def is_point_in_bvc(point, constraints):
    """Check if a point satisfies all constraints of the BVC."""
    if not constraints:
        return True
    for normal, offset in constraints:
        if np.dot(normal, point) < offset:
            return False
    return True


def project_point_to_hyperplane(point, hyperplane):
    """Project a point onto a hyperplane defined by (normal, offset)."""
    normal, offset = hyperplane
    normal_unit = normal / np.linalg.norm(normal)
    distance = np.dot(normal_unit, point) - offset
    projection = point - distance * normal_unit
    return projection


def find_closest_point_in_bvc_shapely(
    robot: Robot,
    all_robots: list[Robot],
    obstacles: list[Obstacle] = None,
    boundary=None,
    use_right_hand_rule=False,
):
    """
    Use shapely to compute the final safe region (BVC) and return the closest point
    on it to the robot's goal.
    """
    bvc_poly = compute_BVC_polygon(
        robot, all_robots, obstacles, boundary, use_right_hand_rule
    )
    goal_pt = Point(robot.goal)

    if bvc_poly and not bvc_poly.is_empty:
        if bvc_poly.contains(goal_pt):
            return robot.goal.copy()
        else:
            # Find nearest point on the polygon to goal
            nearest = nearest_points(goal_pt, bvc_poly)[1]
            return np.array([nearest.x, nearest.y])
    else:
        # Fallback: if safe region is empty, trigger recovery
        print(f"Warning: BVC polygon is empty for robot {robot.id}. Using fallback.")
        return robot.position.copy()


def find_closest_point_in_bvc(goal, position, constraints):
    """
    Find the closest point in the BVC to the goal using the constraint-based approach.
    """
    if not constraints:
        return goal.copy()

    if is_point_in_bvc(goal, constraints):
        return goal.copy()

    # Try projections onto individual constraints
    closest_point = None
    min_distance = float("inf")

    for i, (normal_i, offset_i) in enumerate(constraints):
        projection = project_point_to_hyperplane(goal, (normal_i, offset_i))

        # Check if projection is valid
        other_constraints = [c for j, c in enumerate(constraints) if j != i]
        if is_point_in_bvc(projection, other_constraints):
            distance = np.linalg.norm(projection - goal)
            if distance < min_distance:
                min_distance = distance
                closest_point = projection

    if closest_point is not None:
        return closest_point

    # Try constraint intersections
    for i in range(len(constraints)):
        for j in range(i + 1, len(constraints)):
            normal_i, offset_i = constraints[i]
            normal_j, offset_j = constraints[j]

            # Skip nearly parallel constraints
            cos_angle = np.dot(normal_i, normal_j) / (
                np.linalg.norm(normal_i) * np.linalg.norm(normal_j)
            )
            if abs(cos_angle) > 0.99:
                continue

            # Solve for intersection
            A = np.vstack([normal_i, normal_j])
            b = np.array([offset_i, offset_j])

            try:
                vertex = np.linalg.solve(A, b)
                other_constraints = [
                    c for k, c in enumerate(constraints) if k != i and k != j
                ]

                if is_point_in_bvc(vertex, other_constraints):
                    distance = np.linalg.norm(vertex - goal)
                    if distance < min_distance:
                        min_distance = distance
                        closest_point = vertex
            except np.linalg.LinAlgError:
                continue

    if closest_point is not None:
        return closest_point

    # If all else fails, try average normal direction
    avg_normal = np.zeros(2)
    for normal, _ in constraints:
        avg_normal += normal

    if np.linalg.norm(avg_normal) > 1e-6:
        avg_normal = avg_normal / np.linalg.norm(avg_normal)

        for step in [0.1, 0.2, 0.3, 0.4, 0.5]:
            test_point = position + step * avg_normal
            if is_point_in_bvc(test_point, constraints):
                return test_point

    # Last resort: small random perturbation
    perturb = np.random.rand(2) * 0.05
    return position + perturb


def approximate_bvc_as_polygon(constraints, position, max_radius=10):
    """Approximates the BVC as a polygon for visualization."""
    if not constraints:
        angles = np.linspace(0, 2 * np.pi, 20)
        circle_points = position + max_radius * np.column_stack(
            [np.cos(angles), np.sin(angles)]
        )
        return circle_points

    # Sample points along different directions
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


# -----------------------------------------------------------------------------
# Simulation Functions
# -----------------------------------------------------------------------------
def simulate_bvc_collision_avoidance(
    robots: list,
    dt=0.1,
    max_steps=1000,
    goal_tolerance=0.1,
    use_right_hand_rule=False,
    obstacles=None,
    boundary=None,
    use_shapely=True,
):
    """
    Simulate the BVC collision avoidance with the option to use shapely-based or
    constraint-based approaches.
    """
    for step in range(max_steps):
        # Check for goal completion
        all_reached = True
        for robot in robots:
            if np.linalg.norm(robot.position - robot.goal) > goal_tolerance:
                all_reached = False
                break

        if all_reached:
            print(f"All robots reached their goals in {step} steps!")
            return True

        # Reset collision flags
        for robot in robots:
            robot.in_collision = False

        # Update each robot
        for robot in robots:
            # Skip robots that have already reached their goals
            if np.linalg.norm(robot.position - robot.goal) <= goal_tolerance:
                continue

            # Handle stuck robots
            if robot.stuck_counter > 10:
                # print(f"Robot {robot.id} appears stuck. Initiating recovery.")
                robot.initiate_recovery(obstacles, robots, boundary)
                robot.stuck_counter = 0

            # Compute target point
            if use_shapely:
                target_point = find_closest_point_in_bvc_shapely(
                    robot, robots, obstacles, boundary, use_right_hand_rule
                )
            else:
                bvc_constraints = compute_buffered_voronoi_cell(
                    robot, robots, obstacles, boundary, use_right_hand_rule
                )

                # Handle collision or invalid BVC
                if robot.in_collision or not bvc_constraints:
                    robot.initiate_recovery(obstacles, robots, boundary)
                    recovery_target = robot.position + np.array(
                        [
                            np.cos(robot.recovery_direction),
                            np.sin(robot.recovery_direction),
                        ]
                    )
                    robot.move_to_point(recovery_target, dt, obstacles, boundary)
                    continue

                target_point = find_closest_point_in_bvc(
                    robot.goal, robot.position, bvc_constraints
                )

            # Move robot
            robot.move_to_point(target_point, dt, obstacles, boundary)

    print(
        f"Simulation ended after {max_steps} steps. Not all robots reached their goals."
    )
    return False


# -----------------------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------------------
def visualize_simulation(robots, figure_size=(10, 10), boundary=None, obstacles=None):
    """
    Create a static visualization of the simulation results.
    """
    viz_radius = 0.4
    plt.figure(figsize=figure_size)

    # Draw boundary
    # Draw boundary
    if boundary:
        xmin, xmax, ymin, ymax = boundary
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.plot(
            [xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin],
            "k-",
            linewidth=2,
        )

    # Draw obstacles
    if obstacles:
        for obstacle in obstacles:
            if hasattr(obstacle, "radius"):
                # Draw circles for circular obstacles
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
                # Draw rectangles for rectangular obstacles
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

    # Draw robots and trajectories
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

        # Draw heading indicator
        x_start, y_start = robot.position
        x_end = x_start + viz_radius * np.cos(robot.theta)
        y_end = y_start + viz_radius * np.sin(robot.theta)
        plt.plot([x_start, x_end], [y_start, y_end], color="k", lw=2)

    # Finalize plot
    plt.grid(True)
    plt.legend()
    plt.title("Buffered Voronoi Cell Collision Avoidance Simulation")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.show()


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
    obstacles: list[Obstacle] = None,
    use_shapely=True,
):
    """
    Create an animation of the BVC collision avoidance.
    """
    # Create a copy of robots for simulation
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

    # Setup figure and axes
    fig, ax = plt.subplots(figsize=figure_size)
    if boundary:
        xmin, xmax, ymin, ymax = boundary
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        (boundary_line,) = ax.plot(
            [xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin],
            "k-",
            linewidth=2,
        )

    # Initialize visualization elements
    colors = plt.cm.tab10(np.linspace(0, 1, len(sim_robots)))
    robot_circles = []
    goal_markers = []
    trajectory_lines = []
    bvc_polygons = []
    heading_lines = []

    for i, robot in enumerate(sim_robots):
        # Robot body
        circle = plt.Circle(
            robot.position, viz_radius, fill=True, alpha=0.5, color=colors[i]
        )
        robot_circles.append(ax.add_patch(circle))

        # Goal marker
        goal = ax.plot(robot.goal[0], robot.goal[1], "x", markersize=10, color=colors[i])[
            0
        ]
        goal_markers.append(goal)

        # Trajectory line
        (trajectory,) = ax.plot(
            [], [], "-", linewidth=1.5, color=colors[i], label=f"Robot {robot.id}"
        )
        trajectory_lines.append(trajectory)

        # BVC polygon
        polygon = patches.Polygon(
            np.zeros((1, 2)), closed=True, fill=False, edgecolor=colors[i], alpha=0.3
        )
        bvc_polygons.append(ax.add_patch(polygon))

        # Heading indicator
        x0, y0 = robot.position
        x1 = x0 + viz_radius * np.cos(robot.theta)
        y1 = y0 + viz_radius * np.sin(robot.theta)
        (heading_line,) = ax.plot([x0, x1], [y0, y1], color=colors[i], lw=2)
        heading_lines.append(heading_line)

    # Draw obstacles
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

    # Setup labels and info text
    ax.set_title("Buffered Voronoi Cell Collision Avoidance Animation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.set_aspect("equal")
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, verticalalignment="top")
    performance_text = ax.text(
        0.02, 0.94, "", transform=ax.transAxes, verticalalignment="top", color="green"
    )

    # Run simulation and record positions
    all_positions = []
    step = 0
    all_reached = False
    computation_times = []
    boundary_violations = 0
    obstacle_violations = 0

    while step < max_steps and not all_reached:
        print(f"Step: {step}")
        positions = []
        all_reached = True

        # Reset collision flags
        for robot in sim_robots:
            robot.in_collision = False
            # Make sure we record positions for ALL robots
            positions.append(robot.position.copy())

        # Check if all robots reached goals
        for robot in sim_robots:
            if np.linalg.norm(robot.position - robot.goal) > goal_tolerance:
                all_reached = False
                break

        # Update robot positions
        start_time = time.time()
        for robot in sim_robots:
            # Skip robots that have already reached their goals - they stay in place
            if np.linalg.norm(robot.position - robot.goal) <= goal_tolerance:
                robot.trajectory.append(robot.position.copy())  # Keep trajectory updated
                robot.theta_trajectory.append(robot.theta)  # Keep orientation updated
                continue

            # Check for stuck robots
            if robot.stuck_counter > 20:
                # print(f"Robot {robot.id} appears stuck. Initiating recovery.")
                robot.initiate_recovery(obstacles, sim_robots, boundary)
                robot.stuck_counter = 0

            # Compute target point
            if use_shapely:
                target_point = find_closest_point_in_bvc_shapely(
                    robot, sim_robots, obstacles, boundary, use_right_hand_rule
                )
            else:
                bvc_constraints = compute_buffered_voronoi_cell(
                    robot, sim_robots, obstacles, boundary, use_right_hand_rule
                )

                if robot.in_collision or not bvc_constraints:
                    robot.initiate_recovery(obstacles, sim_robots, boundary)
                    recovery_target = robot.position + np.array(
                        [
                            np.cos(robot.recovery_direction),
                            np.sin(robot.recovery_direction),
                        ]
                    )
                    robot.move_to_point(recovery_target, dt, obstacles, boundary)
                    continue

                target_point = find_closest_point_in_bvc(
                    robot.goal, robot.position, bvc_constraints
                )

            # Check for violations
            if boundary:
                xmin, xmax, ymin, ymax = boundary
                margin = robot.safety_radius * 0.5
                if (
                    robot.position[0] < xmin + margin
                    or robot.position[0] > xmax - margin
                    or robot.position[1] < ymin + margin
                    or robot.position[1] > ymax - margin
                ):
                    boundary_violations += 1

            if obstacles:
                for obstacle in obstacles:
                    if (
                        obstacle.is_point_inside(robot.position)
                        or obstacle.distance_to_point(robot.position)
                        < robot.safety_radius
                    ):
                        obstacle_violations += 1
                        break

            # Move robot
            robot.move_to_point(target_point, dt, obstacles, boundary)

        # Record simulation state
        computation_time = time.time() - start_time
        computation_times.append(computation_time)
        all_positions.append((positions, all_reached, step, computation_time))
        step += 1

    # Animation update function
    def update(frame):
        positions, reached, current_step, comp_time = all_positions[frame]
        for i, robot in enumerate(sim_robots):
            # Use frame index safely by checking the length of trajectory
            if frame < len(robot.trajectory):
                # Update robot position
                robot_circles[i].center = robot.trajectory[frame]

                # Update trajectory
                x_data = [pos[0] for pos in robot.trajectory[: frame + 1]]
                y_data = [pos[1] for pos in robot.trajectory[: frame + 1]]
                trajectory_lines[i].set_data(x_data, y_data)

                # Update heading indicator
                pos_frame = robot.trajectory[frame]
                theta_frame = robot.theta_trajectory[frame]
                x0, y0 = pos_frame
                x1 = x0 + viz_radius * np.cos(theta_frame)
                y1 = y0 + viz_radius * np.sin(theta_frame)
                heading_lines[i].set_data([x0, x1], [y0, y1])

        # Update status text
        status = "COMPLETE" if reached else "IN PROGRESS"
        info_text.set_text(f"Step: {current_step} | Status: {status}")
        performance_text.set_text(
            f"Comp time: {comp_time:.4f}s | Violations: B:{boundary_violations} O:{obstacle_violations}"
        )

        return (
            robot_circles
            + trajectory_lines
            + bvc_polygons
            + heading_lines
            + [info_text, performance_text]
        )

    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(all_positions), interval=interval, blit=True
    )

    # Save animation if requested
    if save_animation:
        anim.save("bvc_collision_avoidance.mp4", writer="ffmpeg", fps=30)

    # Print statistics
    avg_time = np.mean(computation_times)
    max_time = np.max(computation_times)
    print(f"\nPerformance statistics:")
    print(f"  Average computation time: {avg_time:.4f}s")
    print(f"  Maximum computation time: {max_time:.4f}s")
    print(f"  Boundary violations: {boundary_violations}")
    print(f"  Obstacle violations: {obstacle_violations}")

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Environment Setup Functions
# -----------------------------------------------------------------------------
def load_environment(yaml_file):
    """Load environment configuration from YAML file."""
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_environment_from_yaml(
    yaml_file, robot_radius=0.5, max_speed=0.8 * GLOBAL_SCALE
):
    """
    Create robots and obstacles from a YAML configuration file.

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
    for obs_config in config.get("obstacles", []):
        if "radius" in obs_config:
            # Circular obstacle
            center = obs_config["center"]
            radius = obs_config["radius"]
            obstacle = CircleObstacle(center, radius)
        else:
            # Rectangular obstacle
            center = obs_config["center"]
            width = obs_config["width"]
            height = obs_config["height"]
            obstacle = Obstacle(center, width, height)
        obstacles.append(obstacle)

    environment_size = (40, 40)
    return robots, obstacles, environment_size


def run_yaml_environment(
    yaml_config, use_right_hand_rule=True, max_steps=1000, dt=0.05, use_shapely=True
):
    """
    Run a simulation from a YAML configuration.
    """
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
        use_shapely=use_shapely,
    )


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
def main():
    """Parse command line arguments and run the simulation."""
    parser = argparse.ArgumentParser(
        description="BVC Collision Avoidance Simulation with Obstacles"
    )

    # Environment options
    parser.add_argument(
        "--obstacle_type",
        type=str,
        default="rectangle",
        choices=["circle", "rectangle", "free"],
        help="Type of obstacles in the environment",
    )
    parser.add_argument(
        "--num_robots", type=int, default=20, help="Number of robots in the simulation"
    )
    parser.add_argument(
        "--map_density", type=int, default=15, help="Density of obstacles in the map"
    )
    parser.add_argument(
        "--instance_id", type=int, default=2, help="Instance ID for the map"
    )

    # Algorithm options
    parser.add_argument(
        "--use_shapely",
        action="store_true",
        help="Use Shapely for BVC computation (more accurate but slower)",
    )
    parser.add_argument(
        "--no_right_hand_rule",
        action="store_true",
        help="Disable the right-hand rule for deadlock avoidance",
    )

    # Simulation options
    parser.add_argument("--dt", type=float, default=0.05, help="Simulation time step")
    parser.add_argument(
        "--max_steps", type=int, default=2000, help="Maximum simulation steps"
    )
    parser.add_argument(
        "--save_animation",
        action="store_true",
        help="Save the animation to a file",
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the YAML configuration file (overrides other options)",
    )

    args = parser.parse_args()

    if args.config:
        yaml_config = load_environment(args.config)
    else:
        # Generate path based on parameters
        if args.obstacle_type == "circle":
            map_path = f"benchmark_wj/circle_maps_wj/CircleEnv{args.map_density}/agents{args.num_robots}/CircleEnv_{args.map_density}_{args.num_robots}_{args.instance_id}.yaml"
        elif args.obstacle_type == "rectangle":
            map_path = f"benchmark_wj/rect_maps_wj/RectEnv{args.map_density}/agents{args.num_robots}/RectEnv_{args.map_density}_{args.num_robots}_{args.instance_id}.yaml"
        else:
            map_path = f"benchmark_wj/Free_maps_wj/agents{args.num_robots}/Free_0_{args.num_robots}_{args.instance_id}.yaml"

        print(f"Loading map from: {map_path}")
        yaml_config = load_environment(map_path)

    run_yaml_environment(
        yaml_config,
        use_right_hand_rule=not args.no_right_hand_rule,
        max_steps=args.max_steps,
        dt=args.dt,
        use_shapely=args.use_shapely,
    )


if __name__ == "__main__":
    main()
