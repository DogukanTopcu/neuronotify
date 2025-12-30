"""
plotting.py - Specialized Visualization for NeuroNotify Experiments

This module contains dedicated plotting functions for generating the specific
figures required for the research paper (Experiment 2 and Experiment 3).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

def plot_exp2_heatmaps(
    agent: Any,
    profiles: List[Any],
    profile_names: List[str] = ["Student", "Worker", "Workaholic"],
    save_path: str = "exp2_heatmaps_all_personas.png"
) -> plt.Figure:
    """
    Experiment 2: Behavioral heatmaps for three user personas.
    
    Generates a 3x1 grid of heatmaps showing Q-values for action 'Send'.
    Dimensions: Hour of Day (x) vs Recency (y).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Constants
    HOURS = np.arange(24)
    RECENCIES = np.arange(0, 12)  # 0 to 11 hours
    MAX_RECENCY = 48.0
    MAX_ANNOYANCE = 10.0
    
    for idx, (profile, name, ax) in enumerate(zip(profiles, profile_names, axes)):
        q_matrix = np.zeros((len(RECENCIES), len(HOURS)))
        
        for r_idx, recency in enumerate(RECENCIES):
            for h_idx, hour in enumerate(HOURS):
                # Construct State for Exp 2 (7 dimensions: [h, d, r, a, ohe_1, ohe_2, ohe_3])
                norm_hour = hour / 23.0
                norm_day = 0.0  # Assume day 0
                norm_recency = min(recency, MAX_RECENCY) / MAX_RECENCY
                norm_annoyance = 0.0 # Assume low annoyance
                
                # OHE
                user_ohe = np.zeros(len(profiles), dtype=np.float32)
                user_ohe[idx] = 1.0
                
                state = np.array([norm_hour, norm_day, norm_recency, norm_annoyance, *user_ohe], dtype=np.float32)
                
                # Get Q-value for action 1 (Send)
                with np.testing.suppress_warnings() as sup:
                    # Suppress potential warnings from PyTorch/Numpy interop
                    q_values = agent.get_q_values(state)
                    q_matrix[r_idx, h_idx] = q_values[1]
        
        # Plot Heatmap
        sns.heatmap(
            q_matrix, 
            ax=ax, 
            cmap="RdYlGn", 
            cbar=True, # (idx == 2), # Cbar on last one only? No, maybe all for clarity or constrained
            linewidths=0.5,
            xticklabels=2,
            yticklabels=2,
            vmin=np.min(q_matrix),
            vmax=np.max(q_matrix)
        )
        
        ax.set_title(f"{name}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Hour of Day")
        if idx == 0:
            ax.set_ylabel("Hours Since Last Send")
        else:
            ax.set_ylabel("")
            
        ax.invert_yaxis() # 0 at bottom
        
        # Custom ticks
        ax.set_xticks(np.arange(0, 24, 4) + 0.5)
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 4)], rotation=45)

    plt.suptitle("Experiment 2: Learned Policy Heatmaps (Action: Send)", fontsize=16, y=1.05)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
        
    return fig

def plot_exp3_baseline_comparison(
    rewards: Dict[str, float],
    save_path: str = "exp3_baseline_comparison.png"
) -> plt.Figure:
    """
    Experiment 3: Baseline comparison bar chart.
    
    Compares: Random, OptimalStatic, DQN (OHE), Oracle.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(rewards.keys())
    values = list(rewards.values())
    colors = ['gray', 'steelblue', 'green', 'gold']
    
    # Ensure there are enough colors if more baselines
    if len(values) > len(colors):
        colors = plt.cm.tab10(np.linspace(0, 1, len(values)))

    bars = ax.bar(names, values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add values on top
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    ax.set_ylabel("Average Reward per Episode", fontsize=12)
    ax.set_title("Experiment 3: Performance Comparison vs Baselines", fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add multipliers relative to Random
    if len(values) > 0:
        base_val = values[0]
        for i in range(1, len(values)):
            if base_val != 0:
                multiplier = values[i] / base_val
                # Place text inside bar if positive, or above
                y_pos = values[i]/2 if values[i] > 0 else values[i] - 1
                color = 'white' if values[i] > 5 else 'black'
                ax.text(i, y_pos, f"{multiplier:.1f}x", 
                        ha='center', color=color, fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
        
    return fig

def plot_exp3_ohe_vs_pomdp(
    ohe_rewards: List[float],
    pomdp_rewards: List[float],
    title: str = "Experiment 3: OHE vs POMDP Learning Curves",
    save_path: str = "exp3_ohe_vs_pomdp.png"
) -> plt.Figure:
    """
    Experiment 3: Learning curve comparison.
    """
    def smooth(x, window=50):
        if len(x) < window: return x
        return np.convolve(x, np.ones(window)/window, mode='valid')

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot smoothed curves
    w = 50
    ax.plot(smooth(ohe_rewards, w), label='DQN with User ID (OHE)', color='green', linewidth=2)
    ax.plot(smooth(pomdp_rewards, w), label='DQN without User ID (POMDP)', color='orange', linewidth=2, linestyle='--')
    
    # Add raw transparent signal
    ax.plot(ohe_rewards, alpha=0.1, color='green')
    ax.plot(pomdp_rewards, alpha=0.1, color='orange')
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Annotate final values
    if len(ohe_rewards) > 100:
        final_ohe = np.mean(ohe_rewards[-100:])
        ax.annotate(f"OHE: {final_ohe:.1f}", 
                    xy=(len(ohe_rewards), final_ohe), 
                    xytext=(10, 0), textcoords='offset points', color='green', fontweight='bold')

    if len(pomdp_rewards) > 100:
        final_pomdp = np.mean(pomdp_rewards[-100:])
        ax.annotate(f"POMDP: {final_pomdp:.1f}", 
                    xy=(len(pomdp_rewards), final_pomdp), 
                    xytext=(10, 0), textcoords='offset points', color='orange', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved learning curves to {save_path}")
        
    return fig


def plot_exp3_sensitivity_tradeoff(
    results: pd.DataFrame,
    save_path: str = "exp3_sensitivity_tradeoff.png"
) -> plt.Figure:
    """
    Experiment 3: Penalty sensitivity tradeoff.
    
    Dual-axis plot:
    X: Penalty Value (e.g. -1, -3, -5, -10)
    Y1: Sends per Episode (Bar)
    Y2: Click-Through Rate (Line)
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    penalties = results['penalty'].astype(str)
    sends = results['sends_per_episode']
    ctr = results['ctr'] * 100 # Convert to percentage
    
    # Bar plot for Volume (Sends)
    ax1.set_xlabel('Penalty for Ignore (Negative Reward)', fontsize=12)
    ax1.set_ylabel('Avg Sends per Week', fontsize=12, color='steelblue')
    bars = ax1.bar(penalties, sends, color='steelblue', alpha=0.6, label='Volume')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    # Line plot for Quality (CTR)
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Click-Through Rate (%)', fontsize=12, color='crimson')
    line = ax2.plot(penalties, ctr, color='crimson', marker='o', linewidth=3, label='Precision')
    ax2.tick_params(axis='y', labelcolor='crimson')
    
    # Title
    plt.title("Experiment 3: Quantity vs Quality Trade-off", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved sensitivity plot to {save_path}")
        
    return fig

def plot_exp3_sensitivity_curves(
    results: Dict[float, List[float]],
    save_path: str = "exp3_sensitivity_curves.png"
) -> plt.Figure:
    """
    Experiment 3: Learning curves for different penalty values.
    """
    def smooth(x, window=50):
        if len(x) < window: return x
        return np.convolve(x, np.ones(window)/window, mode='valid')

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    for (penalty, rewards), color in zip(sorted(results.items(), reverse=True), colors):
        smoothed = smooth(rewards, 50)
        ax.plot(smoothed, label=f'Penalty {penalty}', color=color, linewidth=2)
        ax.plot(rewards, alpha=0.1, color=color)
        
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Experiment 3: Reward Sensitivity Analysis", fontsize=14, fontweight='bold')
    ax.legend(title="Ignore Penalty")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved sensitivity curves to {save_path}")
        
    return fig
