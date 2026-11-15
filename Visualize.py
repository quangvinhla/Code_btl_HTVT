"""
visualize_results.py
"""
import matplotlib
matplotlib.use("Agg")
import pickle
import os
import matplotlib.pyplot as plt
# ==============================
# CONFIG
# ==============================
RESULT_FILES = {
    25: "./out/simulation25_results.pkl",
    50: "./out/simulation50_results.pkl",
    75: "./out/simulation75_results.pkl",
    100: "./out/simulation100_results.pkl",
}
METHODS = ["Dijkstra", "GA", "Hybrid_PSO_GA"]
OUTPUT_DIR = "./out"
# ==============================
# LOAD RESULTS
# ==============================
def load_results():
    data = {m: {"delay": [], "energy": [], "pdr": []} for m in METHODS}
    nodes_sorted = sorted(RESULT_FILES.keys())

    for n in nodes_sorted:
        path = RESULT_FILES[n]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing result file: {path}")

        with open(path, "rb") as f:
            summary = pickle.load(f)

        for m in METHODS:
            data[m]["delay"].append(summary["methods"][m]["avg_delay"])
            data[m]["energy"].append(summary["methods"][m]["avg_energy"])
            data[m]["pdr"].append(summary["methods"][m]["avg_pdr"])

    return nodes_sorted, data

# ==============================
# PLOTTING
# ==============================
def plot_metric(nodes, data, metric, ylabel, filename):
    plt.figure(figsize=(7, 5))

    for method in METHODS:
        plt.plot(nodes, data[method][metric], marker='o', label=method)

    plt.xlabel("Number of nodes")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Number of Nodes")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


# ==============================
# MAIN
# ==============================
def main():
    print("Visualizing QoS comparison for 25–50–75–100 nodes...\n")

    nodes, data = load_results()

    plot_metric(nodes, data, "delay", "Average End-to-End Delay", "delay_vs_nodes.png")
    plot_metric(nodes, data, "energy", "Average Energy Consumption", "energy_vs_nodes.png")
    plot_metric(nodes, data, "pdr", "Average Packet Delivery Ratio", "pdr_vs_nodes.png")

    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
