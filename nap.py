import random
import time
import matplotlib.pyplot as plt

def noahs_ark_greedy(taxa, costs, pd_gain, budget):
    """
    Greedy algorithm for the Noah's Ark Problem.
    taxa     : list of species names
    costs    : conservation costs for each species
    pd_gain  : PD (phylogenetic diversity) gain for each species
    budget   : total available budget
    """
    n = len(taxa)
    selected = []
    total_cost = 0
    total_pd = 0

    # Compute PD gain per unit cost ratio
    ratio = [(pd_gain[i] / costs[i], i) for i in range(n)]
    ratio.sort(reverse=True)  # Sort in descending order of cost-benefit ratio

    for r, i in ratio:
        if total_cost + costs[i] <= budget:
            selected.append(taxa[i])
            total_cost += costs[i]
            total_pd += pd_gain[i]
        else:
            break

    return selected, total_cost, total_pd


def runtime_experiment():
    """
    Run the greedy algorithm for increasing input sizes (up to n=1000000)
    and measure execution time for each.
    """
    input_sizes = [100,1000,10000,100000,1000000]
    runtimes = []

    for n in input_sizes:
        taxa = [f"T{i}" for i in range(n)]
        costs = [random.uniform(1, 10) for _ in range(n)]
        pd_gain = [random.uniform(10, 100) for _ in range(n)]
        budget = sum(costs) * 0.3  # budget = 30% of total cost

        start_time = time.time()
        noahs_ark_greedy(taxa, costs, pd_gain, budget)
        end_time = time.time()

        runtime = end_time - start_time
        runtimes.append(runtime)
        print(f"n = {n:5d} | runtime = {runtime:.6f} seconds")

    # Plot runtime vs input size
    plt.figure(figsize=(8, 5))
    plt.plot(input_sizes, runtimes, marker='o', linestyle='-', label="Measured runtime")
    plt.title("Runtime of Greedy Algorithm for Noah's Ark Problem (up to n=10,000)")
    plt.xlabel("Number of Taxa (n)")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    runtime_experiment()