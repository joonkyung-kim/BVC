import yaml
from shapely.geometry import box, Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
import random


def create_rectangle(center, width, height):
    x, y = center
    return box(x - width / 2, y - height / 2, x + width / 2, y + height / 2)


def check_target_distances(point, targets, min_target_distance):
    """
    점들 간의 최소 거리를 확인하는 함수
    """
    for target in targets:
        if point.distance(target) < min_target_distance:
            return False
    return True


def check_obstacle_distance(point, obstacles, min_obstacle_distance):
    """
    장애물과의 최소 거리를 확인하는 함수
    """
    for obstacle in obstacles:
        if point.distance(obstacle) < min_obstacle_distance:
            return False
    return True


def is_valid_position(
    point,
    existing_points,
    obstacles,
    min_target_distance,
    min_obstacle_distance,
    map_size=40,
):
    """
    위치가 유효한지 확인하는 함수
    """
    # 기존 target들과의 거리 확인
    if not check_target_distances(point, existing_points, min_target_distance):
        return False

    # 장애물과의 거리 확인
    if not check_obstacle_distance(point, obstacles, min_obstacle_distance):
        return False

    # 맵 경계와의 거리 확인
    boundary_margin = min(min_target_distance, min_obstacle_distance) / 2
    if (
        point.x < boundary_margin
        or point.x > map_size - boundary_margin
        or point.y < boundary_margin
        or point.y > map_size - boundary_margin
    ):
        return False

    return True


def sample_positions(
    num_positions,
    free_area,
    obstacles,
    min_target_distance,
    min_obstacle_distance,
    map_size=40,
    max_attempts=1000,
):
    """
    유효한 위치를 샘플링하는 함수
    """
    positions = []
    attempts = 0

    while len(positions) < num_positions and attempts < max_attempts:
        x = random.uniform(min_obstacle_distance, map_size - min_obstacle_distance)
        y = random.uniform(min_obstacle_distance, map_size - min_obstacle_distance)
        candidate = Point(x, y)

        if free_area.contains(candidate) and is_valid_position(
            candidate,
            positions,
            obstacles,
            min_target_distance,
            min_obstacle_distance,
            map_size,
        ):
            positions.append(candidate)
            attempts = 0  # 성공시 attempts 리셋
        else:
            attempts += 1

    if len(positions) < num_positions:
        raise ValueError(
            f"Could not find valid positions for all {num_positions} points after {max_attempts} attempts"
        )

    return positions


def generate_unique_colors(n):
    return np.random.rand(n, 3)


def save_to_yaml(obstacles, start_positions, goal_positions, num_agents, file_path):
    yaml_data = {
        "obstacles": [
            {
                "center": [
                    float(obstacle.bounds[0] + obstacle.bounds[2]) / 2,
                    float(obstacle.bounds[1] + obstacle.bounds[3]) / 2,
                ],
                "height": float(obstacle.bounds[3] - obstacle.bounds[1]),
                "width": float(obstacle.bounds[2] - obstacle.bounds[0]),
            }
            for obstacle in obstacles
        ],
        "agentNum": num_agents,
        "startPoints": [[float(point.x), float(point.y)] for point in start_positions],
        "goalPoints": [[float(point.x), float(point.y)] for point in goal_positions],
    }

    with open(file_path, "w") as file:
        yaml.dump(yaml_data, file, default_flow_style=False)


def generate_mapf_environment(
    num_agents,
    map_density,
    min_target_distance=2.0,  # target들 간의 최소 거리
    min_obstacle_distance=1.0,  # obstacle과 target 간의 최소 거리
    map_size=40,
    env_name="RectEnv",
    visualize=True,
):
    """
    MAPF 환경을 생성하는 메인 함수
    """
    current_file = os.path.dirname(__file__)
    folder_name = f"../maps/rect_maps/density_{map_density}"
    folder_path = os.path.join(current_file, folder_name)

    file_list = os.listdir(folder_path)
    file_list.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    save_folder_name = f"{env_name}{int(map_density)}"
    save_folder_path = os.path.join(current_file, save_folder_name)
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    for map_id, map_file in enumerate(file_list):
        map_path = os.path.join(folder_path, map_file)
        print(f"Processing map: {map_path}")

        with open(map_path, "r") as file:
            data = yaml.safe_load(file)

        # Create obstacles
        obstacles = []
        for obstacle in data["obstacles"]:
            center = obstacle["center"]
            width = obstacle["width"]
            height = obstacle["height"]
            rect = create_rectangle(center, width, height)
            obstacles.append(rect)

        # Create map boundary and free area
        map_boundary = box(0, 0, map_size, map_size)
        obstacles_union = unary_union(obstacles)
        free_area = map_boundary.difference(obstacles_union)

        try:
            # Sample start and goal positions with specified distances
            start_positions = sample_positions(
                num_agents,
                free_area,
                obstacles,
                min_target_distance,
                min_obstacle_distance,
                map_size,
            )
            goal_positions = sample_positions(
                num_agents,
                free_area,
                obstacles,
                min_target_distance,
                min_obstacle_distance,
                map_size,
            )

            if visualize:
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 10))

                # Plot obstacles
                for obstacle in obstacles:
                    patch = patches.Polygon(
                        list(obstacle.exterior.coords),
                        closed=True,
                        color="red",
                        alpha=0.5,
                    )
                    ax.add_patch(patch)

                # Plot start and goal positions
                colors = generate_unique_colors(num_agents)
                for i, (start, goal, color) in enumerate(
                    zip(start_positions, goal_positions, colors)
                ):
                    ax.plot(
                        start.x,
                        start.y,
                        "o",
                        color=color,
                        markersize=8,
                        label=f"Agent {i+1} Start",
                    )
                    ax.plot(
                        goal.x,
                        goal.y,
                        "x",
                        color=color,
                        markersize=8,
                        label=f"Agent {i+1} Goal",
                    )

                ax.set_xlim(0, map_size)
                ax.set_ylim(0, map_size)
                ax.set_aspect("equal")
                plt.title(
                    f"MAPF Environment\n(Agents: {num_agents}, Density: {map_density}%)"
                )
                plt.grid(True)
                plt.show()

            # Save to YAML
            agent_folder = f"agents{num_agents}"
            agent_folder_path = os.path.join(save_folder_path, agent_folder)
            if not os.path.exists(agent_folder_path):
                os.makedirs(agent_folder_path)

            file_name = f"{env_name}_{map_density}_{num_agents}_{map_id}.yaml"
            save_file_path = os.path.join(agent_folder_path, file_name)
            save_to_yaml(
                obstacles, start_positions, goal_positions, num_agents, save_file_path
            )
            print(f"Successfully saved to {save_file_path}")

        except ValueError as e:
            print(f"Error processing map {map_id}: {e}")
            continue


if __name__ == "__main__":
    # Configuration
    config = {
        "num_agents": 80,
        "map_density": 15,
        "min_target_distance": 2.3,  # target들 간의 최소 거리
        "min_obstacle_distance": 2.0,  # obstacle과 target 간의 최소 거리
        "map_size": 40,
        "env_name": "RectEnv",
        "visualize": False,
    }

    # Generate environment
    generate_mapf_environment(**config)
