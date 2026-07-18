"""
plot_hp_results.py — Load hp search results from pickle and
generate convergence/objective/evaluations plots.
"""

import matplotlib.pyplot as plt
from skopt import load
from skopt.plots import plot_convergence, plot_objective, plot_evaluations

result = load("hp_optim_results.pkl")

best_hyperparameters = {
    "Learning Rate": result.x[0],
    "Hidden": result.x[1],
    "Batch": result.x[2],
    "Layers": result.x[3],
    "Dropout": result.x[4],
    "FFN Expansion": result.x[5],
}

print("Best Hyperparameters:")
for param, value in best_hyperparameters.items():
    print(f"{param}: {value}")

with open('best_hyperparameters.txt', 'w') as f:
    f.write("Best Hyperparameters:\n")
    for param, value in best_hyperparameters.items():
        f.write(f"{param}: {value}\n")

param_names = [
    "Learning Rate", "Hidden", "Batch",
    "Layers", "Dropout", "FFN Expansion",
]

plt.figure(figsize=(12, 8))
plot_convergence(result)
plt.xlabel(r"Number of calls, n", fontsize=16)
plt.ylabel("Min f(x) after n calls", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Convergence Plot", fontsize=20)
plt.savefig('outputs/imgs/convergence_plot.png', dpi=300)
plt.close()

plt.figure(figsize=(12, 8))
plot_objective(result, dimensions=param_names, size=5, n_points=250, levels=30)
plt.savefig('outputs/imgs/objective_plot.png', dpi=300)
plt.close()

plt.figure(figsize=(25, 25))
plot_evaluations(result, dimensions=param_names, size=5)
plt.savefig('outputs/imgs/evaluations_plot.png', dpi=300)
plt.close()
