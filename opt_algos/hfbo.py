import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Colors for each algorithm
COLORS = {
    'SMAC': 'red',
    'RandomSearch': 'blue',
    'GridSearch': 'green',
    'HMF-Opt': 'orange'  # Added HMF-Opt
}

def create_optimization_plot(results, metric, metric_name, seeds, hmf_results_dir=None):
    """Create a plot showing optimization progress for a given metric across all seeds"""
    plt.figure(figsize=(12, 8))
    
    for opt_name in ['SMAC', 'RandomSearch', 'GridSearch', 'HMF-Opt']:
        if opt_name == 'HMF-Opt' and hmf_results_dir:
            # Load and process HMF-Opt results
            all_runs_values = []
            for seed in seeds:
                hmf_file = os.path.join(hmf_results_dir, f"hmf_perf_history_seed{seed}_metric{metric}.npy")
                if not os.path.exists(hmf_file):
                    print(f"Warning: HMF-Opt file {hmf_file} not found, skipping.")
                    continue
                perf_history = np.load(hmf_file)
                costs = perf_history[:, 0]  # Cumulative FLOPS
                values = perf_history[:, 1]  # Best observed values
                # Interpolate to a common x-axis
                common_x = np.linspace(0, max(costs), 1000)
                interp_values = np.interp(common_x, costs, values)
                all_runs_values.append(interp_values)
            if not all_runs_values:
                continue
            all_runs_array = np.array(all_runs_values)
            mean_values = np.mean(all_runs_array, axis=0)
            std_values = np.std(all_runs_array, axis=0)
            x = common_x
        else:
            # Existing code for other optimizers
            all_runs_values = []
            for seed in seeds:
                if (opt_name, metric, seed) not in results:
                    continue
                df = results[(opt_name, metric, seed)]['history']
                if isinstance(df['history'].iloc[0], str):
                    df['history'] = df['history'].str.split('|').apply(lambda x: [tuple(map(float, i.split(':'))) for i in x])
                values = []
                if metric > 7:  # Accuracy: maximize
                    current_best = float('-inf')
                    for row in df.itertuples():
                        for _, val in row.history:
                            val = -val  # Convert back to positive accuracy
                            current_best = max(current_best, val)
                            values.append(current_best)
                else:  # Cross-entropy: minimize
                    current_best = float('inf')
                    for row in df.itertuples():
                        for _, val in row.history:
                            current_best = min(current_best, val)
                            values.append(current_best)
                all_runs_values.append(values)
            if not all_runs_values:
                continue
            max_len = max(len(run) for run in all_runs_values)
            padded_runs = [run + [run[-1]] * (max_len - len(run)) if len(run) < max_len else run for run in all_runs_values]
            all_runs_array = np.array(padded_runs)
            mean_values = np.mean(all_runs_array, axis=0)
            std_values = np.std(all_runs_array, axis=0)
            x = np.arange(len(mean_values))
        
        # Plotting
        plt.plot(x, mean_values, 
                color=COLORS[opt_name],
                label=opt_name,
                linewidth=2)
        plt.fill_between(x, 
                        mean_values - std_values,
                        mean_values + std_values,
                        color=COLORS[opt_name],
                        alpha=0.2)
    
    plt.title(f'Algorithm Comparison - {metric_name}\nSolid lines show mean best value across {len(seeds)} seeds')
    plt.xlabel('Number of Evaluations (Normalized FLOPS)')
    if metric <= 7:
        plt.ylabel('Cross Entropy (lower is better)')
        plt.yscale('log')
    else:
        plt.ylabel('Accuracy (higher is better)')
        plt.yscale('linear')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    return plt.gcf()

def main():
    parser = argparse.ArgumentParser(description='Create optimization plots from saved results')
    parser.add_argument('results_dir', type=str, help='Path to results directory (e.g., results/20250129_023617)')
    parser.add_argument('--hmf-results-dir', type=str, help='Directory containing HMF-Opt results')
    parser.add_argument('--output-dir', type=str, help='Output directory for plots (defaults to results_dir/plots)')
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    hmf_results_dir = os.path.abspath(args.hmf_results_dir) if args.hmf_results_dir else None
    
    results_file = os.path.join(results_dir, "data", "optimization_results.pickle")
    if not os.path.exists(results_file):
        print(f"Error: Could not find results file at {results_file}")
        sys.exit(1)

    with open(results_file, 'rb') as f:
        data = pickle.load(f)

    all_results = data['all_results']
    config = data['config']
    
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from: {results_file}")
    if hmf_results_dir:
        print(f"Loading HMF-Opt results from: {hmf_results_dir}")
    print(f"Saving plots to: {output_dir}")

    for metric in config['metrics']:
        metric_name = config['metric_names'][metric]
        print(f"Creating plot for {metric_name}...")
        fig = create_optimization_plot(all_results, metric, metric_name, config['seeds'], hmf_results_dir)
        output_file = os.path.join(output_dir, f"optimization_plot_{metric_name.replace(' ', '_')}.png")
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot to {output_file}")

    print("\nAll plots have been created successfully!")

if __name__ == "__main__":
    main()