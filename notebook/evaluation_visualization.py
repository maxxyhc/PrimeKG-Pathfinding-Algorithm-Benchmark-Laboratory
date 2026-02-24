# ============================================
# EVALUATION VISUALIZATION
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def create_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Create a nicely formatted summary table with best values highlighted.
    
    Args:
        summary: DataFrame from generate_summary()
    
    Returns:
        Styled DataFrame for display
    """
    # Define which metrics are "lower is better"
    lower_is_better = ['hub_node_ratio', 'edit_distance']
    
    # Create a copy for formatting
    display_df = summary.copy()
    
    # Drop n_pathways for cleaner display
    if 'n_pathways' in display_df.columns:
        n_pathways = display_df['n_pathways'].iloc[0]
        display_df = display_df.drop(columns=['n_pathways'])
    
    return display_df


def plot_comparison_heatmap(summary: pd.DataFrame, save_path: str = None):
    """
    Create a heatmap comparing all algorithms across all metrics.
    
    Args:
        summary: DataFrame from generate_summary()
        save_path: Optional path to save the figure
    """
    # Prepare data
    lower_is_better = ['hub_node_ratio', 'edit_distance']
    
    df = summary.copy()
    if 'n_pathways' in df.columns:
        df = df.drop(columns=['n_pathways'])
    
    # Normalize data for coloring (0-1 scale, higher = better)
    normalized = df.copy()
    for col in df.columns:
        if col in lower_is_better:
            # Invert so that lower values get higher scores
            normalized[col] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-10)
        else:
            normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-10)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create heatmap
    im = ax.imshow(normalized.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(df.index, fontsize=12)
    
    # Add text annotations
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            value = df.iloc[i, j]
            # Determine text color based on background
            text_color = 'white' if normalized.iloc[i, j] < 0.5 else 'black'
            
            # Add marker for best value
            is_best = False
            if df.columns[j] in lower_is_better:
                is_best = value == df.iloc[:, j].min()
            else:
                is_best = value == df.iloc[:, j].max()
            
            text = f'{value:.4f}'
            if is_best:
                text = f'★ {value:.4f}'
            
            ax.text(j, i, text, ha='center', va='center', 
                   color=text_color, fontsize=10, fontweight='bold' if is_best else 'normal')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Performance (normalized)', rotation=-90, va='bottom', fontsize=11)
    
    # Title and labels
    ax.set_title('Algorithm Comparison Heatmap\n(★ = Best, Green = Better)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add note about lower-is-better metrics
    ax.text(0.5, -0.15, '* hub_node_ratio and edit_distance: lower is better (colors inverted)', 
           transform=ax.transAxes, ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()


def plot_radar_chart(summary: pd.DataFrame, save_path: str = None):
    """
    Create a radar/spider chart comparing algorithms.
    
    Args:
        summary: DataFrame from generate_summary()
        save_path: Optional path to save the figure
    """
    # Select key metrics for radar chart
    metrics = ['precision', 'recall_at_6_hops', 'f1_score', 
               'path_length_accuracy', 'relation_accuracy', 'mrr']
    
    df = summary.copy()
    
    # Filter to available metrics
    metrics = [m for m in metrics if m in df.columns]
    
    # Number of variables
    num_vars = len(metrics)
    
    # Compute angle for each metric
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Colors for each algorithm
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    
    # Plot each algorithm
    for idx, (algo_name, row) in enumerate(df.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=algo_name, color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.1)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)
    
    # Title
    ax.set_title('Algorithm Performance Radar Chart\n(Higher = Better)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()


def plot_metric_bars(summary: pd.DataFrame, save_path: str = None):
    """
    Create grouped bar charts for each metric category.
    
    Args:
        summary: DataFrame from generate_summary()
        save_path: Optional path to save the figure
    """
    df = summary.copy()
    if 'n_pathways' in df.columns:
        df = df.drop(columns=['n_pathways'])
    
    # Define metric groups
    metric_groups = {
        'Node Accuracy': ['precision', 'recall_at_6_hops', 'f1_score'],
        'Path Quality': ['path_length_accuracy', 'mrr'],
        'Structural Similarity': ['edit_distance', 'relation_accuracy'],
        'Hub Avoidance': ['hub_node_ratio']
    }
    
    # Colors for algorithms
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    algo_colors = {algo: colors[i % len(colors)] for i, algo in enumerate(df.index)}
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax_idx, (group_name, metrics) in enumerate(metric_groups.items()):
        ax = axes[ax_idx]
        
        # Filter to available metrics
        metrics = [m for m in metrics if m in df.columns]
        if not metrics:
            continue
        
        x = np.arange(len(metrics))
        width = 0.8 / len(df.index)
        
        for i, (algo_name, row) in enumerate(df.iterrows()):
            values = [row[m] for m in metrics]
            offset = (i - len(df.index) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=algo_name, color=algo_colors[algo_name])
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, rotation=90)
        
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(group_name, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0, 1.3)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Add note for lower-is-better metrics
        lower_better = [m for m in metrics if m in ['hub_node_ratio', 'edit_distance']]
        if lower_better:
            ax.text(0.5, -0.25, f'* {", ".join(lower_better)}: lower is better',
                   transform=ax.transAxes, ha='center', fontsize=8, style='italic', color='gray')
    
    plt.suptitle('Algorithm Comparison by Metric Category', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()


def plot_best_algorithm_summary(summary: pd.DataFrame, save_path: str = None):
    """
    Create a summary chart showing which algorithm is best for each metric.
    
    Args:
        summary: DataFrame from generate_summary()
        save_path: Optional path to save the figure
    """
    lower_is_better = ['hub_node_ratio', 'edit_distance']
    
    df = summary.copy()
    if 'n_pathways' in df.columns:
        df = df.drop(columns=['n_pathways'])
    
    # Find best algorithm for each metric
    best_algos = {}
    best_values = {}
    for col in df.columns:
        if col in lower_is_better:
            best_algos[col] = df[col].idxmin()
            best_values[col] = df[col].min()
        else:
            best_algos[col] = df[col].idxmax()
            best_values[col] = df[col].max()
    
    # Count wins per algorithm
    win_counts = pd.Series(best_algos.values()).value_counts()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Best algorithm per metric
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    algo_colors = {algo: colors[i % len(colors)] for i, algo in enumerate(df.index)}
    
    metrics = list(best_algos.keys())
    y_pos = np.arange(len(metrics))
    
    bar_colors = [algo_colors[best_algos[m]] for m in metrics]
    bars = ax1.barh(y_pos, [best_values[m] for m in metrics], color=bar_colors)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(metrics, fontsize=11)
    ax1.set_xlabel('Best Score', fontsize=11)
    ax1.set_title('Best Algorithm per Metric', fontsize=12, fontweight='bold')
    
    # Add algorithm names on bars
    for i, (bar, metric) in enumerate(zip(bars, metrics)):
        algo = best_algos[metric]
        note = ' (↓)' if metric in lower_is_better else ''
        ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{algo}{note}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax1.set_xlim(0, 1.4)
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Win count per algorithm
    win_colors = [algo_colors[algo] for algo in win_counts.index]
    bars2 = ax2.bar(win_counts.index, win_counts.values, color=win_colors)
    
    ax2.set_ylabel('Number of Metrics Won', fontsize=11)
    ax2.set_title('Algorithm Win Count', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(win_counts.values) + 1)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add legend
    legend_patches = [mpatches.Patch(color=algo_colors[algo], label=algo) for algo in df.index]
    fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              ncol=len(df.index), fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()


def generate_full_report(summary: pd.DataFrame, save_prefix: str = 'algorithm_comparison'):
    """
    Generate a complete visual report with all charts.
    
    Args:
        summary: DataFrame from generate_summary()
        save_prefix: Prefix for saved file names
    """
    print("="*60)
    print("GENERATING VISUAL REPORT")
    print("="*60)
    
    print("\n1. Creating heatmap...")
    plot_comparison_heatmap(summary, f'{save_prefix}_heatmap.png')
    
    print("\n2. Creating radar chart...")
    plot_radar_chart(summary, f'{save_prefix}_radar.png')
    
    print("\n3. Creating bar charts...")
    plot_metric_bars(summary, f'{save_prefix}_bars.png')
    
    print("\n4. Creating best algorithm summary...")
    plot_best_algorithm_summary(summary, f'{save_prefix}_best.png')
    
    print("\n" + "="*60)
    print("REPORT COMPLETE")
    print("="*60)
    print(f"\nSaved files:")
    print(f"  - {save_prefix}_heatmap.png")
    print(f"  - {save_prefix}_radar.png")
    print(f"  - {save_prefix}_bars.png")
    print(f"  - {save_prefix}_best.png")


# Quick display function
def display_summary_table(summary: pd.DataFrame):
    """
    Print a nicely formatted summary table in the terminal.
    """
    lower_is_better = ['hub_node_ratio', 'edit_distance']
    
    df = summary.copy()
    if 'n_pathways' in df.columns:
        df = df.drop(columns=['n_pathways'])
    
    print("\n" + "="*100)
    print("ALGORITHM COMPARISON SUMMARY")
    print("="*100)
    
    # Header
    print(f"\n{'Metric':<25}", end='')
    for algo in df.index:
        print(f"{algo:>18}", end='')
    print(f"{'Best':>15}")
    print("-"*100)
    
    # Rows
    for col in df.columns:
        print(f"{col:<25}", end='')
        
        # Find best
        if col in lower_is_better:
            best_val = df[col].min()
            best_algo = df[col].idxmin()
        else:
            best_val = df[col].max()
            best_algo = df[col].idxmax()
        
        for algo in df.index:
            val = df.loc[algo, col]
            marker = "★" if val == best_val else " "
            print(f"{marker}{val:>16.4f}", end='')
        
        note = " (↓)" if col in lower_is_better else ""
        print(f"{best_algo:>14}{note}")
    
    print("="*100)
    print("\n★ = Best  |  (↓) = Lower is better")


print("✓ Evaluation visualization loaded")
print("  Use: generate_full_report(summary)")
print("  Or:  display_summary_table(summary)")