import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import math
import heapq
from collections import deque

plt.ion()


class LandmarkGraph:

    def __init__(self):

        self.graph = nx.Graph()

        self.positions = {}

        self.landmark_names = {}

        self.districts = {}

    # --------------------------
    # Load landmark nodes
    # --------------------------
    def load_nodes(self, filename):

        df = pd.read_csv(filename)

        for _, row in df.iterrows():

            node_id = row["id"]

            self.graph.add_node(node_id)

            self.positions[node_id] = (
                row["lon"],
                row["lat"]
            )

            self.landmark_names[node_id] = (
                row["name"]
            )

            self.districts[node_id] = (
                row["district"]
            )

    # --------------------------
    # Load edges
    # --------------------------
    def load_edges(self, filename):

        df = pd.read_csv(filename)

        for _, row in df.iterrows():

            source = row["u"]

            target = row["v"]

            weight = self.calculate_distance(
                source,
                target
            )

            self.graph.add_edge(
                source,
                target,
                weight=weight,
                road=row["road"]
            )

    # --------------------------
    # Search landmarks by name
    # --------------------------
    def get_node_by_name(self, search_text):

        search_text = (
            search_text
            .lower()
            .strip()
        )

        for node_id, name in (
            self.landmark_names.items()
        ):

            if search_text in name.lower():

                return node_id

        return None

    # --------------------------
    # Distance calculations
    # --------------------------
    def calculate_distance(
        self,
        node_a,
        node_b
    ):

        x1, y1 = self.positions[node_a]

        x2, y2 = self.positions[node_b]

        return math.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )

    def heuristic(
        self,
        node,
        goal
    ):

        return self.calculate_distance(
            node,
            goal
        )

    # --------------------------
    # Visual overlap fixes
    # --------------------------
    def get_adjusted_positions(self):

        adjusted = {}

        occupied = []

        for node, (x, y) in (
            self.positions.items()
        ):

            new_x = x
            new_y = y

            for ox, oy in occupied:

                if (
                    abs(new_x - ox) < 0.002
                    and
                    abs(new_y - oy) < 0.002
                ):

                    new_x += 0.002
                    new_y += 0.002

            adjusted[node] = (
                new_x,
                new_y
            )

            occupied.append(
                (new_x, new_y)
            )

        return adjusted

    # --------------------------
    # Search colors
    # --------------------------
    def get_depth_color(
        self,
        depth
    ):

        colors = [

            "#ffe6f2",
            "#f5ccff",
            "#d9d9ff",
            "#cceeff",
            "#ccfff2",
            "#e6ffcc",
            "#fff5cc"

        ]

        return colors[
            min(
                depth,
                len(colors) - 1
            )
        ]

    # --------------------------
    # Animate search
    # --------------------------
    def animate_search(
        self,
        depth_map,
        final_path=None
    ):

        plt.clf()

        adjusted_positions = (
            self.get_adjusted_positions()
        )

        node_colors = []

        for node in self.graph.nodes():

            if (
                final_path and
                node in final_path
            ):

                node_colors.append(
                    "red"
                )

            elif node in depth_map:

                node_colors.append(
                    self.get_depth_color(
                        depth_map[node]
                    )
                )

            else:

                node_colors.append(
                    "lightgray"
                )

        nx.draw(
            self.graph,
            pos=adjusted_positions,
            labels={
                n: n
                for n in self.graph.nodes()
            },
            node_color=node_colors,
            node_size=300,
            font_size=6
        )

        plt.title(
            "Search Progress"
        )

        plt.draw()
        plt.pause(0.25)

    # --------------------------
    # Search algorithms
    # --------------------------
    def bfs(
        self,
        start,
        goal
    ):

        queue = deque()

        parent = {}

        depth_map = {}

        queue.append(start)

        depth_map[start] = 0

        while queue:

            current = queue.popleft()

            self.animate_search(
                depth_map
            )

            if current == goal:

                return (
                    self.reconstruct_path(
                        parent,
                        goal
                    ),
                    depth_map
                )

            for neighbor in (
                self.graph.neighbors(
                    current
                )
            ):

                if (
                    neighbor
                    not in depth_map
                ):

                    parent[
                        neighbor
                    ] = current

                    depth_map[
                        neighbor
                    ] = (
                        depth_map[
                            current
                        ] + 1
                    )

                    queue.append(
                        neighbor
                    )

        return None, depth_map

    def dfs(
        self,
        start,
        goal
    ):

        stack = []

        parent = {}

        depth_map = {}

        stack.append(start)

        depth_map[start] = 0

        while stack:

            current = stack.pop()

            self.animate_search(
                depth_map
            )

            if current == goal:

                return (
                    self.reconstruct_path(
                        parent,
                        goal
                    ),
                    depth_map
                )

            for neighbor in (
                self.graph.neighbors(
                    current
                )
            ):

                if (
                    neighbor
                    not in depth_map
                ):

                    parent[
                        neighbor
                    ] = current

                    depth_map[
                        neighbor
                    ] = (
                        depth_map[
                            current
                        ] + 1
                    )

                    stack.append(
                        neighbor
                    )

        return None, depth_map

    def greedy_search(
        self,
        start,
        goal
    ):

        pq = []

        parent = {}

        depth_map = {}

        heapq.heappush(
            pq,
            (
                self.heuristic(
                    start,
                    goal
                ),
                start
            )
        )

        depth_map[start] = 0

        while pq:

            _, current = (
                heapq.heappop(
                    pq
                )
            )

            self.animate_search(
                depth_map
            )

            if current == goal:

                return (
                    self.reconstruct_path(
                        parent,
                        goal
                    ),
                    depth_map
                )

            for neighbor in (
                self.graph.neighbors(
                    current
                )
            ):

                if (
                    neighbor
                    not in depth_map
                ):

                    parent[
                        neighbor
                    ] = current

                    depth_map[
                        neighbor
                    ] = (
                        depth_map[
                            current
                        ] + 1
                    )

                    heapq.heappush(
                        pq,
                        (
                            self.heuristic(
                                neighbor,
                                goal
                            ),
                            neighbor
                        )
                    )

        return None, depth_map

    def a_star(
        self,
        start,
        goal
    ):

        pq = []

        parent = {}

        g_cost = {}

        depth_map = {}

        g_cost[start] = 0

        depth_map[start] = 0

        heapq.heappush(
            pq,
            (
                self.heuristic(
                    start,
                    goal
                ),
                start
            )
        )

        while pq:

            _, current = (
                heapq.heappop(
                    pq
                )
            )

            self.animate_search(
                depth_map
            )

            if current == goal:

                return (
                    self.reconstruct_path(
                        parent,
                        goal
                    ),
                    depth_map
                )

            for neighbor in (
                self.graph.neighbors(
                    current
                )
            ):

                edge_cost = (
                    self.graph[
                        current
                    ][neighbor][
                        "weight"
                    ]
                )

                new_cost = (
                    g_cost[current]
                    + edge_cost
                )

                if (
                    neighbor
                    not in g_cost
                    or
                    new_cost <
                    g_cost[
                        neighbor
                    ]
                ):

                    g_cost[
                        neighbor
                    ] = new_cost

                    parent[
                        neighbor
                    ] = current

                    depth_map[
                        neighbor
                    ] = (
                        depth_map[
                            current
                        ] + 1
                    )

                    f_cost = (
                        new_cost
                        +
                        self.heuristic(
                            neighbor,
                            goal
                        )
                    )

                    heapq.heappush(
                        pq,
                        (
                            f_cost,
                            neighbor
                        )
                    )

        return None, depth_map

    # --------------------------
    # Path reconstruction
    # --------------------------
    def reconstruct_path(
        self,
        parent,
        goal
    ):

        path = []

        current = goal

        while current in parent:

            path.append(
                current
            )

            current = parent[
                current
            ]

        path.append(
            current
        )

        path.reverse()

        return path


# --------------------------
# Benchmark helper
# --------------------------
def benchmark_algorithm(
    graph,
    name,
    function,
    start,
    goal
):

    start_time = (
        time.perf_counter()
    )

    path, depth_map = (
        function(
            start,
            goal
        )
    )

    end_time = (
        time.perf_counter()
    )

    runtime = (
        end_time -
        start_time
    )

    nodes_expanded = len(
        depth_map
    )

    path_length = len(
        path
    )

    efficiency = (
        path_length /
        nodes_expanded
    )

    return {

        "name": name,

        "expanded": (
            nodes_expanded
        ),

        "path_length": (
            path_length
        ),

        "efficiency": (
            efficiency
        ),

        "runtime": runtime
    }


def print_benchmark_table(
    results
):

    print(
        "\n" +
        "=" * 65
    )

    print(
        f"{'ALGORITHM':<12}"
        f"{'EXPANDED':<12}"
        f"{'PATH LEN':<12}"
        f"{'EFFICIENCY':<15}"
        f"{'RUNTIME':<12}"
    )

    print(
        "=" * 65
    )

    for result in results:

        print(

            f"{result['name'].upper():<12}"

            f"{result['expanded']:<12}"

            f"{result['path_length']:<12}"

            f"{result['efficiency']:<15.2f}"

            f"{result['runtime']:<12.4f}"

        )

    print(
        "=" * 65
    )


def main():

    script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    graph = LandmarkGraph()

    graph.load_nodes(
        os.path.join(
            script_dir,
            "kc_landmarks_nodes.csv"
        )
    )

    graph.load_edges(
        os.path.join(
            script_dir,
            "kc_landmarks_edges.csv"
        )
    )

    print(
        "\nExample landmark search:"
    )

    print(
        "Liberty, Union, Plaza..."
    )

    start_name = input(
        "\nStart landmark: "
    )

    goal_name = input(
        "Goal landmark: "
    )

    start = (
        graph.get_node_by_name(
            start_name
        )
    )

    goal = (
        graph.get_node_by_name(
            goal_name
        )
    )

    if (
        start is None
        or
        goal is None
    ):

        print(
            "\nLandmark not found."
        )

        return

    algorithm = input(

        "\nChoose algorithm "
        "(bfs/dfs/greedy/astar/benchmark): "

    ).lower()

    if algorithm == "benchmark":

        results = []

        results.append(
            benchmark_algorithm(
                graph,
                "bfs",
                graph.bfs,
                start,
                goal
            )
        )

        results.append(
            benchmark_algorithm(
                graph,
                "dfs",
                graph.dfs,
                start,
                goal
            )
        )

        results.append(
            benchmark_algorithm(
                graph,
                "greedy",
                graph.greedy_search,
                start,
                goal
            )
        )

        results.append(
            benchmark_algorithm(
                graph,
                "astar",
                graph.a_star,
                start,
                goal
            )
        )

        print_benchmark_table(
            results
        )

        return

    if algorithm == "bfs":

        path, depth_map = (
            graph.bfs(
                start,
                goal
            )
        )

    elif algorithm == "dfs":

        path, depth_map = (
            graph.dfs(
                start,
                goal
            )
        )

    elif algorithm == "greedy":

        path, depth_map = (
            graph.greedy_search(
                start,
                goal
            )
        )

    else:

        path, depth_map = (
            graph.a_star(
                start,
                goal
            )
        )

    graph.animate_search(
        depth_map,
        path
    )

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()