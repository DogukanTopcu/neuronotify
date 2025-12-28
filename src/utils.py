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

from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
    user_id: int = 0,
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
    
    This visualization is critical for the IEEE paper. It shows how the
    learned policy values sending notifications across different hours
    and recency levels. A well-trained agent should show:
    
    - High Q-values (bright) during awake hours
    - Low Q-values (dark) during sleep hours
    - Lower Q-values when recency is low (recently sent)
    
    Args:
        agent: Trained DQNAgent instance
        user_id: User ID to encode in states
        annoyance: Fixed annoyance level for visualization
        day: Day of week (0-6)
        hours_range: Range of hours to plot (start, end)
        recency_range: Range of recency values (start, end)
        title: Plot title
        figsize: Figure dimensions
        save_path: Optional path to save the figure
        show: Whether to display the plot
        cmap: Colormap for the heatmap
        
    Returns:
        Matplotlib Figure object
    """
    # Define grid
    hours = np.arange(hours_range[0], hours_range[1])
    recencies = np.arange(recency_range[0], recency_range[1])
    
    # Create Q-value matrix
    q_matrix = np.zeros((len(recencies), len(hours)))
    
    # Normalization constants (matching env.py)
    MAX_RECENCY = 48.0
    MAX_ANNOYANCE = 10.0
    MAX_USER_ID = 10.0
    
    for i, recency in enumerate(recencies):
        for j, hour in enumerate(hours):
            # Construct normalized state
            norm_hour = hour / 23.0
            norm_day = day / 6.0
            norm_recency = min(recency, MAX_RECENCY) / MAX_RECENCY
            norm_annoyance = min(annoyance, MAX_ANNOYANCE) / MAX_ANNOYANCE
            user_id_encoded = user_id / MAX_USER_ID
            
            state = np.array([
                norm_hour,
                norm_day,
                norm_recency,
                norm_annoyance,
                user_id_encoded
            ], dtype=np.float32)
            
            # Get Q-value for Send action (action=1)
            q_values = agent.get_q_values(state)
            q_matrix[i, j] = q_values[1]  # Q-value for Send
            
    # Create figure with heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(
        q_matrix,
        aspect='auto',
        cmap=cmap,
        origin='lower',
        extent=[hours_range[0], hours_range[1], recency_range[0], recency_range[1]]
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Q-Value (Send)', fontsize=11)
    
    # Styling
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Hours Since Last Notification', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add hour labels
    ax.set_xticks(np.arange(0, 24, 2))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], rotation=45)
    
    # Add recency labels
    ax.set_yticks(np.arange(recency_range[0], recency_range[1], 2))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
    if show:
        plt.show()
        
    return fig


def plot_dual_heatmap(
    agent,
    user_profiles: List[dict],
    annoyance: float = 0.0,
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create side-by-side heatmaps comparing Q-values for different user profiles.
    
    This visualization demonstrates how the same neural network learns
    different scheduling patterns for different users based on the
    user_id encoded in the state vector.
    
    Args:
        agent: Trained DQNAgent instance
        user_profiles: List of dicts with 'user_id', 'name', and optional 'annoyance'
        annoyance: Default annoyance level if not specified per profile
        figsize: Figure dimensions
        save_path: Optional path to save the figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib Figure object
    """
    n_profiles = len(user_profiles)
    fig, axes = plt.subplots(1, n_profiles, figsize=figsize)
    
    if n_profiles == 1:
        axes = [axes]
        
    hours = np.arange(0, 24)
    recencies = np.arange(0, 11)
    
    MAX_RECENCY = 48.0
    MAX_ANNOYANCE = 10.0
    MAX_USER_ID = 10.0
    
    # Compute all Q-matrices for consistent color scaling
    q_matrices = []
    
    for profile in user_profiles:
        user_id = profile['user_id']
        user_annoyance = profile.get('annoyance', annoyance)
        q_matrix = np.zeros((len(recencies), len(hours)))
        
        for i, recency in enumerate(recencies):
            for j, hour in enumerate(hours):
                state = np.array([
                    hour / 23.0,
                    0.0,  # day
                    min(recency, MAX_RECENCY) / MAX_RECENCY,
                    min(user_annoyance, MAX_ANNOYANCE) / MAX_ANNOYANCE,
                    user_id / MAX_USER_ID
                ], dtype=np.float32)
                
                q_values = agent.get_q_values(state)
                q_matrix[i, j] = q_values[1]
                
        q_matrices.append(q_matrix)
        
    # Determine global color scale
    vmin = min(q.min() for q in q_matrices)
    vmax = max(q.max() for q in q_matrices)
    
    # Plot each heatmap
    for idx, (ax, profile, q_matrix) in enumerate(zip(axes, user_profiles, q_matrices)):
        im = ax.imshow(
            q_matrix,
            aspect='auto',
            cmap='RdYlGn',
            origin='lower',
            extent=[0, 24, 0, 11],
            vmin=vmin,
            vmax=vmax
        )
        
        ax.set_xlabel('Hour of Day', fontsize=11)
        ax.set_ylabel('Hours Since Last Notification', fontsize=11)
        ax.set_title(f"{profile['name']}", fontsize=12, fontweight='bold')
        ax.set_xticks(np.arange(0, 24, 4))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 4)], rotation=45)
        
    # Add shared colorbar
    fig.colorbar(im, ax=axes, shrink=0.8, label='Q-Value (Send)')
    
    plt.suptitle('Learned Notification Scheduling Patterns', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
    if show:
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
