"""
iot_graph_full.py
"""
import matplotlib
matplotlib.use("Agg")
from typing import Tuple, Dict, Any, List
import random
import math
import os
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Cấu hình 
DEFAULT_CONFIG = {
    "num_nodes": 30,
    "area_size": 100.0,       
    "tx_range": 30.0,      
    "seed": 42,

    "w_delay": 0.4,
    "w_energy": 0.3,
    "w_pdr": 0.3,
  
    "delay_random_min": 1.0,
    "delay_random_max": 5.0,
    "energy_coef_min": 0.03,   
    "energy_coef_max": 0.12,
    "bandwidth_min": 50.0,     
    "bandwidth_max": 250.0,
    "per_min": 0.005,
    "per_max": 0.15
}

class IoTGraphBuilder:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        random.seed(self.config["seed"]) 
        np.random.seed(self.config["seed"])
        self.G = nx.Graph()

    def generate_positions(self) -> Dict[int, Tuple[float, float]]:
        N = self.config["num_nodes"]
        L = self.config["area_size"]
        positions = {i: (random.uniform(0, L), random.uniform(0, L)) for i in range(N)}
        return positions

    @staticmethod
    def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def build_graph(self) -> nx.Graph:
        pos = self.generate_positions()
        G = nx.Graph()
        for n, p in pos.items():
            G.add_node(n, pos=p)

        tx = self.config["tx_range"]
        for i in range(self.config["num_nodes"]):
            for j in range(i + 1, self.config["num_nodes"]):
                d = self.euclidean(pos[i], pos[j])
                if d <= tx:
                    delay = d / 5.0 + random.uniform(self.config["delay_random_min"], self.config["delay_random_max"]) 
                    coef = random.uniform(self.config["energy_coef_min"], self.config["energy_coef_max"])
                    energy = d * coef
                    bandwidth = max(self.config["bandwidth_min"], self.config["bandwidth_max"] - d * 1.5)
                    per = min(self.config["per_max"], self.config["per_min"] + (d / tx) * self.config["per_max"]) 
                    pdr = max(0.0, 1.0 - per)

                    G.add_edge(i, j, distance=d, delay=delay, energy=energy, bandwidth=bandwidth, per=per, pdr=pdr)

        self.G = G
        return G

    def normalize_edge_attributes(self, attr_names: List[str]) -> Dict[str, Dict[Tuple[int, int], float]]:
        result = {}
        for attr in attr_names:
            vals = [d.get(attr, 0.0) for _, _, d in self.G.edges(data=True)]
            if len(vals) == 0:
                result[attr] = {}
                continue
            vmin, vmax = min(vals), max(vals)
            denom = vmax - vmin if vmax != vmin else 1.0
            normalized = {}
            for u, v, d in self.G.edges(data=True):
                raw = d.get(attr, 0.0)
                normalized[(u, v)] = (raw - vmin) / denom
            result[attr] = normalized
        return result

    def compute_composite_cost(self) -> None:
        weights = self.normalize_edge_attributes(["delay", "energy", "pdr"])
        w_delay = self.config["w_delay"]
        w_energy = self.config["w_energy"]
        w_pdr = self.config["w_pdr"]

        for u, v, d in self.G.edges(data=True):
            ndelay = weights["delay"].get((u, v), 0.0)
            nenergy = weights["energy"].get((u, v), 0.0)
            npdr = weights["pdr"].get((u, v), 0.0)
            cost = w_delay * ndelay + w_energy * nenergy + w_pdr * (1.0 - npdr)
            d["ndelay"] = ndelay
            d["nenergy"] = nenergy
            d["npdr"] = npdr
            d["cost"] = cost

    # ------------------ Utility / I/O ------------------
    def save_graph(self, path: str) -> None:
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(path, "wb") as f:
            pickle.dump(self.G, f)

    def load_graph(self, path: str) -> nx.Graph:
        with open(path, "rb") as f:
            self.G = pickle.load(f)
        return self.G

    def visualize(self, show_edge_labels: bool = False, save_path: str = None) -> None:
        """Vẽ đồ thị IoT với vị trí node và màu cạnh theo cost."""
        if self.G is None or len(self.G) == 0:
            raise RuntimeError("Graph is empty. Build graph first.")

        pos = nx.get_node_attributes(self.G, "pos")
        plt.figure(figsize=(8, 8))
        nodes = nx.draw_networkx_nodes(self.G, pos, node_size=120, node_color="skyblue")
        nx.draw_networkx_labels(self.G, pos, font_size=8)
        costs = [d.get("cost", 0.0) for _, _, d in self.G.edges(data=True)]
        if len(costs) > 0:
            vmin, vmax = min(costs), max(costs)
            denom = vmax - vmin if vmax != vmin else 1.0
            colors = [(c - vmin) / denom for c in costs]
        else:
            colors = [0 for _ in costs]

        edges = nx.draw_networkx_edges(self.G, pos, edge_color=colors, edge_cmap=plt.cm.viridis, width=2)
        plt.colorbar(edges)

        if show_edge_labels:
            edge_labels = {(u, v): f"{d['cost']:.2f}" for u, v, d in self.G.edges(data=True)}
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=6)

        plt.axis("off")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


    # ------------------ Dijkstra & path helpers ------------------
    def dijkstra_path(self, source: int, target: int, weight_attr: str = "cost") -> List[int]:
        if source not in self.G or target not in self.G:
            raise ValueError("Source or target not in graph")
        path = nx.shortest_path(self.G, source=source, target=target, weight=weight_attr)
        return path

    def path_cost(self, path: List[int], attr: str = "cost") -> float:
        total = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            total += self.G[u][v].get(attr, 0.0)
        return total

    def all_pairs_shortest_paths_cost(self, attr: str = "cost") -> Dict[Tuple[int, int], float]:
        length = dict(nx.all_pairs_dijkstra_path_length(self.G, weight=attr))
        result = {}
        for u in length:
            for v, val in length[u].items():
                result[(u, v)] = val
        return result

    # ------------------ Initial population generation ------------------
    def perturb_path(self, path: List[int], max_attempts: int = 10) -> List[int]:
        if len(path) < 4:
            return path.copy()
        for _ in range(max_attempts):
            i = random.randint(0, len(path) - 3)
            j = random.randint(i + 2, len(path) - 1)
            a, b = path[i], path[j]
            try:
                sub = nx.shortest_path(self.G, source=a, target=b, weight="cost")
            except nx.NetworkXNoPath:
                continue
            new_path = path[:i] + sub + path[j + 1:]
            if len(new_path) == len(set(new_path)):
                return new_path
        return path.copy()

    def generate_initial_population(self, source: int, target: int, pop_size: int = 20) -> List[List[int]]:
        pop = []
        try:
            base = self.dijkstra_path(source, target, weight_attr="cost")
        except Exception:
            # fallback: shortest by distance
            base = nx.shortest_path(self.G, source=source, target=target, weight="distance")

        pop.append(base)
        attempts = 0
        while len(pop) < pop_size and attempts < pop_size * 10:
            candidate = self.perturb_path(base)
            if candidate not in pop:
                pop.append(candidate)
            attempts += 1

        if len(pop) < pop_size:
            nodes = list(self.G.nodes())
            while len(pop) < pop_size:
                try:
                    rpath = nx.shortest_path(self.G, source=source, target=target)
                    if rpath not in pop:
                        pop.append(rpath)
                except nx.NetworkXNoPath:
                    break
        return pop
    def get_reasonable_source_target(self, percentile=0.8):
        source = 0
        pos = nx.get_node_attributes(self.G, "pos")

        distances = []
        for node, coord in pos.items():
            if node == source:
                continue
            d = math.dist(pos[source], coord)
            distances.append((d, node))
        distances.sort(key=lambda x: x[0])
        k = int(len(distances) * percentile)
        candidate_nodes = distances[k:]

        _, target = random.choice(candidate_nodes)

        return source, target


# ----------------------------
# CLI / Example usage
# ----------------------------
if __name__ == "__main__":
    cfg = {
        "num_nodes": 75,
        "area_size": 120.0,
        "tx_range": 30.0,
        "seed": 123,
        "w_delay": 0.4,
        "w_energy": 0.35,
        "w_pdr": 0.25
    }

    builder = IoTGraphBuilder(cfg)
    G = builder.build_graph()
    builder.compute_composite_cost()
    source, target = builder.get_reasonable_source_target(percentile=0.8)

    print(f"Fixed source-target: {source} -> {target}")

    with open("./out/source_target.txt", "w") as f:
        f.write(f"{source},{target}")

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    builder.save_graph("./out/iot_graph75.gpickle")

    builder.visualize(show_edge_labels=True, save_path="./out/iot_graph75.png")

    nodes = list(G.nodes())
    if len(nodes) >= 2:
        path = builder.dijkstra_path(source, target)
        cost = builder.path_cost(path)
        print(f"Dijkstra path {source} -> {target}: {path}, cost={cost:.4f}")

        pop = builder.generate_initial_population(source, target, pop_size=12)
        print("Initial population (paths):")
        for p in pop:
            print(p)

        print("Done.")
