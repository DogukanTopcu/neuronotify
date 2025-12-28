"""
utils.py - Metrics Logging and Visualization Helpers

This module provides utilities for tracking training metrics and visualizing
the learned policy behavior. Key features include:

1. MetricsLogger: Track and store training statistics
2. Learning Curve Plotting: Visualize reward progression
3. Behavioral Heatmap: Analyze learned Q-values across state space

These visualizations are critical for the IEEE paper to demonstrate that
the DQN has learned meaningful behavioral patterns aligned with user personas.
"""

from typing import List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.networks import get_device


def compute_moving_average(data: List[float], window_size: int = 50) -> np.ndarray:
    """
    Compute the moving average of a data series.
    
    The moving average smooths noisy training curves to reveal underlying
    trends in learning performance.
    
    Args:
        data: List of values (e.g., episode rewards)
        window_size: Size of the averaging window
        
    Returns:
        Numpy array of smoothed values
    """
    if len(data) < window_size:
        # Not enough data points; return cumulative average
        return np.cumsum(data) / np.arange(1, len(data) + 1)
        
    # Compute convolution-based moving average
    kernel = np.ones(window_size) / window_size
    padded = np.pad(data, (window_size - 1, 0), mode='edge')
    moving_avg = np.convolve(padded, kernel, mode='valid')
    
    return moving_avg


def plot_learning_curve(
    rewards: List[float],
    window_size: int = 50,
    title: str = "DQN Learning Curve",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot the learning curve showing reward progression over episodes.
    
    Displays both raw episode rewards and a smoothed moving average
    to visualize learning progress despite episode-to-episode variance.
    
    Args:
        rewards: List of total rewards per episode
        window_size: Size of moving average window
        title: Plot title
        figsize: Figure dimensions (width, height)
        save_path: Optional path to save the figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    episodes = np.arange(len(rewards))
    moving_avg = compute_moving_average(rewards, window_size)
    
    # Plot raw rewards with transparency
    ax.plot(
        episodes, 
        rewards, 
        alpha=0.3, 
        color='steelblue',
        label='Episode Reward'
    )
    
    # Plot moving average
    ax.plot(
        episodes, 
        moving_avg, 
        color='darkblue',
        linewidth=2,
        label=f'Moving Average (window={window_size})'
    )
    
    # Styling
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
    if show:
        plt.show()
        
    return fig


def plot_behavioral_heatmap(
    agent,
    user_profile: Optional[Any] = None,
    num_users: int = 2,
    annoyance: float = 0.0,
    day: int = 0,
    hours_range: Tuple[int, int] = (0, 24),
    recency_range: Tuple[int, int] = (0, 11),
    title: str = "Q-Value Heatmap: Action=Send",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
    show: bool = True,
    cmap: str = "RdYlGn"
) -> plt.Figure:
    """
    Create a behavioral heatmap showing Q-values for the Send action.
    """
    # Define grid
    hours = np.arange(hours_range[0], hours_range[1])
    recencies = np.arange(recency_range[0], recency_range[1])
    
    # Create Q-value matrix
    q_matrix = np.zeros((len(recencies), len(hours)))
    
    # Normalization constants
    MAX_RECENCY = 48.0
    MAX_ANNOYANCE = 10.0
    
    for i, recency in enumerate(recencies):
        for j, hour in enumerate(hours):
            norm_hour = hour / 23.0
            norm_day = day / 6.0
            norm_recency = min(recency, MAX_RECENCY) / MAX_RECENCY
            norm_annoyance = min(annoyance, MAX_ANNOYANCE) / MAX_ANNOYANCE
            
            # Behavioral features (defaults for heatmap)
            # In a real scenario, these could be parameters, but for generic heatmap 
            # we assume the user is awake and not working to see peak potential.
            is_working = 0.0
            is_awake = 1.0
            
            # Construct state based on feature count
            base_features = [norm_hour, norm_day, norm_recency, norm_annoyance]
            
            if num_users > 0:
                # OHE mode: [base, is_working, is_awake, user_ohe]
                user_ohe = np.zeros(num_users, dtype=np.float32)
                if user_profile and 0 <= user_profile.user_id < num_users:
                    user_ohe[user_profile.user_id] = 1.0
                state = np.concatenate([base_features, [is_working, is_awake], user_ohe]).astype(np.float32)
            else:
                # POMDP mode: [base, is_working, is_awake]
                state = np.array(base_features + [is_working, is_awake], dtype=np.float32)
            
            # Compatibility check for older environments (Exp 1/2) if needed
            # But here we assume the new structure
            
            q_values = agent.get_q_values(state)
            q_matrix[i, j] = q_values[1]
            
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        q_matrix,
        aspect='auto',
        cmap=cmap,
        origin='lower',
        extent=[hours_range[0], hours_range[1], recency_range[0], recency_range[1]]
    )
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Q-Value (Send)', fontsize=11)
    
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Hours Since Last Notification', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(np.arange(0, 24, 2))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], rotation=45)
    ax.set_yticks(np.arange(recency_range[0], recency_range[1], 2))
    
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show: plt.show()
    return fig


# Alias for backward compatibility and Experiment 3
plot_policy_heatmap = plot_behavioral_heatmap


def evaluate_agent(
    agent,
    env,
    n_episodes: int = 10,
    epsilon: float = 0.0,
    user_profiles: Optional[List[Any]] = None
) -> float:
    """
    Evaluate an agent over multiple episodes.

    Args:
        agent: The agent to evaluate
        env: The environment
        n_episodes: Number of episodes to run
        epsilon: Exploration rate during evaluation (default 0.0 for pure exploitation)
        user_profiles: Optional list of profiles to sample from randomly per episode

    Returns:
        Average reward per episode
    """
    total_rewards = []

    for _ in range(n_episodes):
        options = {}
        if user_profiles:
            options["user_profile"] = np.random.choice(user_profiles)

        if hasattr(agent, "reset_hidden"):
            agent.reset_hidden()

        state, _ = env.reset(options=options)
        episode_reward = 0.0
        done = False

        while not done:
            # Most agents have an act method
            action = agent.act(state, epsilon=epsilon)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    return float(np.mean(total_rewards))


def plot_dual_heatmap(
    agent,
    profile1: Any,
    profile2: Any,
    num_users: int = 2,
    title1: str = "Profile 1",
    title2: str = "Profile 2",
    figsize: Tuple[int, int] = (16, 6),
    **kwargs
) -> plt.Figure:
    """
    Compare two user personas side-by-side using heatmaps.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    plt.sca(ax1)
    plot_behavioral_heatmap(agent, user_profile=profile1, num_users=num_users, title=title1, show=False, **kwargs)
    
    plt.sca(ax2)
    plot_behavioral_heatmap(agent, user_profile=profile2, num_users=num_users, title=title2, show=False, **kwargs)
    
    plt.tight_layout()
    plt.show()
    return fig


class MetricsLogger:
    """
    Logger for tracking training metrics over time.
    
    Stores episode-level and step-level statistics for analysis
    and visualization. Useful for monitoring training progress
    and debugging learning issues.
    """
    
    def __init__(self):
        """Initialize empty metric containers."""
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_clicks: List[int] = []
        self.episode_sends: List[int] = []
        self.episode_churns: List[bool] = []
        self.losses: List[float] = []
        self.epsilons: List[float] = []
        
    def log_episode(
        self,
        reward: float,
        length: int,
        clicks: int = 0,
        sends: int = 0,
        churned: bool = False
    ) -> None:
        """
        Log metrics for a completed episode.
        
        Args:
            reward: Total episode reward
            length: Episode length (steps)
            clicks: Number of successful clicks
            sends: Total notifications sent
            churned: Whether the episode ended in churn
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_clicks.append(clicks)
        self.episode_sends.append(sends)
        self.episode_churns.append(churned)
        
    def log_training_step(self, loss: float, epsilon: float) -> None:
        """
        Log metrics from a training step.
        
        Args:
            loss: Training loss value
            epsilon: Current exploration rate
        """
        self.losses.append(loss)
        self.epsilons.append(epsilon)
        
    def get_summary(self, last_n: int = 100) -> dict:
        """
        Get summary statistics for recent episodes.
        
        Args:
            last_n: Number of recent episodes to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        recent_rewards = self.episode_rewards[-last_n:]
        recent_lengths = self.episode_lengths[-last_n:]
        recent_churns = self.episode_churns[-last_n:]
        
        return {
            'mean_reward': np.mean(recent_rewards) if recent_rewards else 0,
            'std_reward': np.std(recent_rewards) if recent_rewards else 0,
            'mean_length': np.mean(recent_lengths) if recent_lengths else 0,
            'churn_rate': np.mean(recent_churns) if recent_churns else 0,
            'total_episodes': len(self.episode_rewards),
            'total_training_steps': len(self.losses)
        }
        
    def plot_training_summary(
        self,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Create a comprehensive training summary visualization.
        
        Args:
            figsize: Figure dimensions
            save_path: Optional path to save the figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        episodes = np.arange(len(self.episode_rewards))
        
        # Plot 1: Episode Rewards
        ax1 = axes[0, 0]
        ax1.plot(episodes, self.episode_rewards, alpha=0.3, color='steelblue')
        if len(self.episode_rewards) > 50:
            ma = compute_moving_average(self.episode_rewards, 50)
            ax1.plot(episodes, ma, color='darkblue', linewidth=2, label='MA-50')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Episode Lengths
        ax2 = axes[0, 1]
        ax2.plot(episodes, self.episode_lengths, alpha=0.5, color='green')
        ax2.axhline(y=168, color='red', linestyle='--', label='Max (168 hours)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Episode Lengths')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Training Loss
        ax3 = axes[1, 0]
        if self.losses:
            steps = np.arange(len(self.losses))
            ax3.plot(steps, self.losses, alpha=0.3, color='red')
            if len(self.losses) > 100:
                ma_loss = compute_moving_average(self.losses, 100)
                ax3.plot(steps, ma_loss, color='darkred', linewidth=2)
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Loss')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Epsilon Decay
        ax4 = axes[1, 1]
        if self.epsilons:
            ax4.plot(self.epsilons, color='purple')
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Epsilon')
        ax4.set_title('Exploration Rate')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            
        if show:
            plt.show()
            
        return fig
