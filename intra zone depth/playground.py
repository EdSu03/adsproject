import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from itertools import product
import colorsys
import networkx as nx


class TripCoordinateEstimation:
    def __init__(self, num_zones=4):
        self.width = int(200 * (num_zones**0.5))
        self.height = int(200 * (num_zones**0.5))
        self.num_zones = num_zones
        self.num_horiz_roads = 4 * num_zones
        self.num_vert_roads = 4 * num_zones

        self.zones = self.generate_random_zones()

        # Initialize street graph
        self.street_graph = nx.Graph()
        self.generate_street_network()

        self.time_per_unit = 10
        self.traffic_function = lambda: 0.9 + random.random() * 0.2

        self.trip_count = 20
        self.trips = self.generate_trips(1)
        self.show_estimated = False
        self.show_real = True
        self.selected_trip = None

    def generate_names(self):
        prefixes = [
            "North",
            "South",
            "East",
            "West",
            "Central",
            "Lower",
            "Upper",
            "Old",
            "New",
            "Inner",
            "Outer",
            "Upper",
            "Lower",
            "Little",
            "Big",
            "Saint",
            "King",
            "Queen",
            "Prince",
            "Princess",
            "Royal",
            "Grand",
        ]
        suffixes = [
            "District",
            "Quarter",
            "Heights",
            "Village",
            "Park",
            "Square",
            "Gardens",
            "Town",
            "City",
            "Vista",
            "Valley",
            "Hills",
            "Meadows",
            "Forest",
            "Grove",
            "Lake",
            "River",
            "Bay",
            "Harbor",
            "Port",
            "Beach",
            "Cove",
        ]

        names = [f"{prefix} {suffix}" for prefix, suffix in product(prefixes, suffixes)]
        random.shuffle(names)
        return names[: self.num_zones]

    def generate_colours(self):
        colours = []
        for i in range(self.num_zones):
            hue = i / self.num_zones
            lightness = 0.5
            saturation = 0.9
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            hex_colour = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )
            colours.append(hex_colour)
        random.shuffle(colours)
        return colours

    def generate_random_zones(self):
        min_size = min(self.width, self.height) // (self.num_zones // 2)

        def split_area(x, y, width, height, remaining_zones):
            if remaining_zones <= 1 or min(width, height) < min_size:
                return [{"x": x, "y": y, "width": width, "height": height}]

            split_vertically = width > height
            split_range = (width if split_vertically else height) // 3
            split_pos = random.randint(split_range, 2 * split_range)

            first_half = remaining_zones // 2
            second_half = remaining_zones - first_half

            if split_vertically:
                return split_area(x, y, split_pos, height, first_half) + split_area(
                    x + split_pos, y, width - split_pos, height, second_half
                )
            else:
                return split_area(x, y, width, split_pos, first_half) + split_area(
                    x, y + split_pos, width, height - split_pos, second_half
                )

        base_zones = split_area(0, 0, self.width, self.height, self.num_zones)
        return [
            {
                "id": i,
                "x": z["x"],
                "y": z["y"],
                "width": z["width"],
                "height": z["height"],
                "name": name,
                "colour": colour,
            }
            for i, (z, name, colour) in enumerate(
                zip(base_zones, self.generate_names(), self.generate_colours())
            )
        ]

    def generate_street_network(self):
        horiz_spacing = self.width // self.num_horiz_roads
        vert_spacing = self.height // self.num_vert_roads

        # Create grid nodes
        for i in range(self.num_horiz_roads + 1):
            for j in range(self.num_vert_roads + 1):
                x = i * horiz_spacing
                y = j * vert_spacing
                node_id = f"{x},{y}"
                self.street_graph.add_node(node_id, x=x, y=y)

        # Connect horizontally
        for i in range(self.num_horiz_roads):
            for j in range(self.num_vert_roads + 1):
                x1 = i * horiz_spacing
                x2 = (i + 1) * horiz_spacing
                y = j * vert_spacing
                node1 = f"{x1},{y}"
                node2 = f"{x2},{y}"
                distance = horiz_spacing
                self.street_graph.add_edge(node1, node2, weight=distance)

        # Connect vertically
        for i in range(self.num_horiz_roads + 1):
            for j in range(self.num_vert_roads):
                x = i * horiz_spacing
                y1 = j * vert_spacing
                y2 = (j + 1) * vert_spacing
                node1 = f"{x},{y1}"
                node2 = f"{x},{y2}"
                distance = vert_spacing
                self.street_graph.add_edge(node1, node2, weight=distance)

        # Remove 5% of edges randomly
        edges = list(self.street_graph.edges())
        num_to_remove = int(len(edges) * 0.05)
        edges_to_remove = random.sample(edges, num_to_remove)
        self.street_graph.remove_edges_from(edges_to_remove)

    def is_point_in_zone(self, point, zone):
        x, y = point
        return (
            x >= zone["x"]
            and x <= zone["x"] + zone["width"]
            and y >= zone["y"]
            and y <= zone["y"] + zone["height"]
        )

    def get_street_points_in_zone(self, zone_id):
        zone = next((z for z in self.zones if z["id"] == zone_id), None)
        if not zone:
            return []

        points_in_zone = []
        for node_id, data in self.street_graph.nodes(data=True):
            if self.is_point_in_zone((data["x"], data["y"]), zone):
                points_in_zone.append({"x": data["x"], "y": data["y"], "id": node_id})

        return points_in_zone


    def generate_trips(self, num_trips):
        trips = []

        for i in range(num_trips):
            start_zone_index = random.randint(0, len(self.zones) - 1)
            start_zone = self.zones[start_zone_index]

            end_zone_index = random.randint(0, len(self.zones) - 1)
            end_zone = self.zones[end_zone_index]

            start_point = random.choice(
                self.get_street_points_in_zone(start_zone["id"])
            )
            end_point = random.choice(self.get_street_points_in_zone(end_zone["id"]))

            distance = nx.shortest_path_length(
                self.street_graph, start_point["id"], end_point["id"], weight="weight"
            )
            base_time = distance / self.time_per_unit
            travel_time = self.traffic_function() * base_time

            trips.append(
                {
                    "id": i,
                    "start_zone_id": start_zone["id"],
                    "start_zone_name": start_zone["name"],
                    "end_zone_id": end_zone["id"],
                    "end_zone_name": end_zone["name"],
                    "travel_time": travel_time,
                    "real_start_point": dict(start_point),
                    "real_end_point": dict(end_point),
                    "estimated_start_point": None,
                    "estimated_end_point": None,
                }
            )

        return trips

    def visualize(self):
        plt.figure(figsize=(15, 10))

        # Background
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.gca().set_facecolor("#f8f9fa")

        # Draw the street network
        for u, v in self.street_graph.edges():
            u_data = self.street_graph.nodes[u]
            v_data = self.street_graph.nodes[v]
            plt.plot(
                [u_data["x"], v_data["x"]],
                [u_data["y"], v_data["y"]],
                color="#777777",
                linewidth=1,
            )

        # Draw all zones
        for zone in self.zones:
            rect = patches.Rectangle(
                (zone["x"], zone["y"]),
                zone["width"],
                zone["height"],
                linewidth=1,
                edgecolor="#333333",
                facecolor=zone["colour"],
                alpha=0.5,
            )
            plt.gca().add_patch(rect)
            plt.text(
                zone["x"] + zone["width"] / 2,
                zone["y"] + zone["height"] / 2,
                zone["name"],
                ha="center",
                va="center",
                fontweight="bold",
            )

        # TODO
        # Draw the predicted path along the street network
        #     Versus the actual path along the street network

        plt.tight_layout()
        plt.show()

    def run(self):
        self.trips = self.approximate_trips_locations()

        # TODO: Show all trips rather than just one
        self.selected_trip = self.trips[0]

        self.visualize()

    # TODO
    def approximate_trips_locations(self):
        return []


# Example usage
if __name__ == "__main__":
    app = TripCoordinateEstimation(num_zones=8)
    app.run()
