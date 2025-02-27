import yaml
import os
from shapely.geometry import Point, box
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import time


# Function to create a circle using center and radius
def create_circle(center, radius):
    x, y = center
    return Point(x, y).buffer(radius)


# Function to create a rectangle using center, width, and height
def create_rectangle(center, width, height):
    x, y = center
    return box(x - width / 2, y - height / 2, x + width / 2, y + height / 2)


def is_valid_position(point, existing_points, obstacles, min_distance=2, map_size=40):
    # Check distance from existing points
    for existing_point in existing_points:
        if point.distance(existing_point) < min_distance:
            return False

    # Check distance from each obstacle
    for obstacle in obstacles:
        if point.distance(obstacle) < 1.0:
            return False

    # Check distance from the boundary of the map
    if (
        point.x < min_distance
        or point.x > map_size - min_distance
        or point.y < min_distance
        or point.y > map_size - min_distance
    ):
        return False

    return True


def sample_positions(num_positions, free_area, obstacles, min_distance=1, map_size=40):
    positions = []
    while len(positions) < num_positions:
        x, y = random.uniform(min_distance, map_size - min_distance), random.uniform(
            min_distance, map_size - min_distance
        )
        candidate = Point(x, y)
        if free_area.contains(candidate) and is_valid_position(
            candidate,
            positions,
            obstacles,
            map_size=map_size,
            min_distance=min_distance,
        ):
            positions.append(candidate)
    return positions


# Function to generate unique colors
def generate_unique_colors(n):
    # np.random.seed(1233)  # For reproducible colors
    return np.random.rand(n, 3)  # Generate n unique RGB colors


def save_to_yaml(obstacles, start_positions, goal_positions, num_agents, file_path):
    # Convert circle (Point buffer) objects to Python native types for serialization
    yaml_data = {
        "obstacles": [
            {
                "center": [float(obstacle.centroid.x), float(obstacle.centroid.y)],
                "radius": float(
                    obstacle.bounds[2] - obstacle.centroid.x
                ),  # Assuming obstacle is a circle
            }
            for obstacle in obstacles
        ],
        "agentNum": num_agents,
        "startPoints": [[float(point.x), float(point.y)] for point in start_positions],
        "goalPoints": [[float(point.x), float(point.y)] for point in goal_positions],
    }

    # Writing to a YAML file
    with open(file_path, "w") as file:
        yaml.dump(yaml_data, file, default_flow_style=False)


if __name__ == "__main__":
    # Load every YAML file in the folder

    # Define the number of agents
    num_agents = 20
    map_density = 15
    env_name = "CircleEnv"
    map_id = 0

    # Get the current working directory``
    cwd = os.getcwd()
    current_file = os.path.dirname(__file__)
    # folder_name = f"density_{map_density}"

    # map folder
    # folder_name = f"CircleEnv{map_density}/maps"
    folder_name = f"../maps/circle_maps/density_{map_density}"
    folder_path = os.path.join(current_file, folder_name)

    # Get the list of files in the folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_list = [
        f for f in os.listdir(folder_path) if f.endswith(".yaml") or f.endswith(".yml")
    ]
    print(f"file list: {len(file_list)}")

    save_folder_name = f"{env_name}{int(map_density)}"
    save_folder_path = os.path.join(current_file, save_folder_name)
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    print("file list: ", file_list)
    for map_file in file_list:
        map_path = os.path.join(folder_path, map_file)
        print(f"map path: {map_path}")
        if not os.path.exists(map_path):
            os.makedirs(map_path)
        with open(map_path, "r") as file:
            data = yaml.safe_load(file)
        # print(f"data: {data}")

        obstacles = []

        # Process each circular obstacle and add to the list
        for obstacle in data["obstacles"]:
            center = obstacle["center"]
            radius = obstacle["radius"]
            circle = create_circle(center, radius)
            obstacles.append(circle)

        # Create a large rectangle representing the entire map
        map_boundary = box(0, 0, 40, 40)  # Adjust the size as per your map

        # Create a combined shape of all obstacles
        obstacles_union = unary_union(obstacles)

        # Area where agents can be placed
        free_area = map_boundary.difference(obstacles_union)

        # Sample start and goal positions
        start_positions = sample_positions(
            num_agents, free_area, obstacles, map_size=40, min_distance=1.5
        )
        goal_positions = sample_positions(
            num_agents, free_area, obstacles, map_size=40, min_distance=2.1
        )

        # Now you have 20 pairs of start and goal positions
        agent_positions = list(zip(start_positions, goal_positions))

        # Generate colors for each agent
        colors = generate_unique_colors(num_agents)

        # Visualization with unique colors for each agent pair
        fig, ax = plt.subplots()

        # Plot obstacles as before
        for obstacle in obstacles:
            patch = patches.Polygon(
                list(obstacle.exterior.coords), closed=True, color="red"
            )
            ax.add_patch(patch)

        # Plot start and goal positions with unique colors
        for i in range(num_agents):
            start = start_positions[i]
            goal = goal_positions[i]
            color = colors[i]
            ax.plot(
                start.x, start.y, "o", color=color, label=f"Agent {i+1} Start"
            )  # Start position
            ax.plot(
                goal.x, goal.y, "x", color=color, label=f"Agent {i+1} Goal"
            )  # Goal position

        # Set plot limits and show the plot
        ax.set_xlim(0, 40)  # Adjust as per your map size
        ax.set_ylim(0, 40)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title(f"MAPF Environment (# of agents {num_agents})")
        # plt.legend()
        # plt.show()
        plt.close()

        agent_folder = f"agents{num_agents}"
        agent_folder_path = os.path.join(save_folder_path, agent_folder)
        if not os.path.exists(agent_folder_path):
            os.makedirs(agent_folder_path)

        file_name = f"{env_name}_{map_density}_{int(num_agents)}_{map_id}.yaml"
        save_file_path = os.path.join(agent_folder_path, file_name)
        print(f"Data saved to {save_file_path}")
        save_to_yaml(
            obstacles, start_positions, goal_positions, num_agents, save_file_path
        )

        map_id += 1
