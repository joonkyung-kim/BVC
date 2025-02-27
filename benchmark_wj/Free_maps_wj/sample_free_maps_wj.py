import yaml
from shapely.geometry import Point, box
import random
import os
from typing import List, Tuple


class MapfEnvironmentGenerator:
    def __init__(
        self,
        map_size: float = 40.0,
        agent_size: float = 1.0,
        min_start_distance: float = 2.0,
        min_wall_distance: float = 1.0,
        min_target_distance: float = 2.0,
    ):
        self.map_size = map_size
        self.agent_size = agent_size
        self.min_start_distance = min_start_distance
        self.min_wall_distance = min_wall_distance
        self.min_target_distance = min_target_distance

        # Create map boundary
        self.map_boundary = box(0, 0, map_size, map_size)

    def is_valid_start_position(
        self, point: Point, existing_points: List[Point]
    ) -> bool:
        """Check if a start position is valid."""
        # Check distance from map boundaries
        if (
            point.x < self.min_wall_distance
            or point.x > self.map_size - self.min_wall_distance
            or point.y < self.min_wall_distance
            or point.y > self.map_size - self.min_wall_distance
        ):
            return False

        # Check distance from other start positions
        for existing_point in existing_points:
            if point.distance(existing_point) < self.min_start_distance:
                return False

        return True

    def is_valid_target_position(
        self, point: Point, start_point: Point, existing_targets: List[Point]
    ) -> bool:
        """Check if a target position is valid."""
        # Check distance from map boundaries
        if (
            point.x < self.min_wall_distance
            or point.x > self.map_size - self.min_wall_distance
            or point.y < self.min_wall_distance
            or point.y > self.map_size - self.min_wall_distance
        ):
            return False

        # Check minimum distance from start position
        if point.distance(start_point) < self.min_target_distance:
            return False

        # Check distance from other target positions
        for existing_target in existing_targets:
            if point.distance(existing_target) < self.min_target_distance:
                return False

        return True

    def generate_positions(self, num_agents: int) -> Tuple[List[Point], List[Point]]:
        """Generate valid start and target positions for all agents."""
        start_positions = []
        target_positions = []

        for _ in range(num_agents):
            # Generate start position
            while True:
                x = random.uniform(
                    self.min_wall_distance, self.map_size - self.min_wall_distance
                )
                y = random.uniform(
                    self.min_wall_distance, self.map_size - self.min_wall_distance
                )
                start_point = Point(x, y)

                if self.is_valid_start_position(start_point, start_positions):
                    start_positions.append(start_point)
                    break

            # Generate target position
            while True:
                x = random.uniform(
                    self.min_wall_distance, self.map_size - self.min_wall_distance
                )
                y = random.uniform(
                    self.min_wall_distance, self.map_size - self.min_wall_distance
                )
                target_point = Point(x, y)

                if self.is_valid_target_position(
                    target_point, start_point, target_positions
                ):
                    target_positions.append(target_point)
                    break

        return start_positions, target_positions

    def save_to_yaml(
        self,
        start_positions: List[Point],
        target_positions: List[Point],
        num_agents: int,
        file_path: str,
    ):
        """Save the environment configuration to a YAML file."""
        yaml_data = {
            "map_size": float(self.map_size),
            "agent_size": float(self.agent_size),
            "agentNum": num_agents,
            "startPoints": [
                [float(point.x), float(point.y)] for point in start_positions
            ],
            "goalPoints": [
                [float(point.x), float(point.y)] for point in target_positions
            ],
        }

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write to YAML file
        with open(file_path, "w") as file:
            yaml.dump(yaml_data, file, default_flow_style=False)
        print("Saved environment configuration to", file_path)


def main():
    # Example usage
    generator = MapfEnvironmentGenerator(
        map_size=40.0,
        agent_size=1.0,  # diameter
        min_start_distance=1.5,  # center to center
        min_wall_distance=2.3,
        min_target_distance=2.3,
    )

    # Generate environment for 10 agents
    num_agents = 20
    for i in range(20):
        start_positions, target_positions = generator.generate_positions(num_agents)
        # Save to YAML file
        generator.save_to_yaml(
            start_positions,
            target_positions,
            num_agents,
            f"/mnt/Topics/Learning/CBF/map_generator/Free_maps_wj/agents{num_agents}/Free_0_{num_agents}_{i}.yaml",
        )


if __name__ == "__main__":
    main()
