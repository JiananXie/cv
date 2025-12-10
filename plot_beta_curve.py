import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results(csv_file):
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found. Please run the experiment script first.")
        return

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    if df.empty:
        print("Error: CSV file is empty.")
        return

    # Sort by beta just in case
    df = df.sort_values('beta')

    plt.figure(figsize=(12, 5))
    
    # Plot Position Error
    plt.subplot(1, 2, 1)
    plt.plot(df['beta'], df['pos_error'], marker='o', linestyle='-', linewidth=2)
    plt.xscale('log')
    plt.xlabel(r'$\beta$ (Log Scale)', fontsize=12)
    plt.ylabel('Median Position Error (m)', fontsize=12)
    plt.title('Effect of $\\beta$ on Position Error', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    # Plot Orientation Error
    plt.subplot(1, 2, 2)
    plt.plot(df['beta'], df['ori_error'], marker='o', linestyle='-', linewidth=2, color='orange')
    plt.xscale('log')
    plt.xlabel(r'$\beta$ (Log Scale)', fontsize=12)
    plt.ylabel('Median Orientation Error (deg)', fontsize=12)
    plt.title('Effect of $\\beta$ on Orientation Error', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.tight_layout()
    output_file = 'beta_performance_curve.png'
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    plot_results('beta_experiment_results.csv')
