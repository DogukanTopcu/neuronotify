"""
NeuroNotify: DQN-Based Push Notification Optimization

A Deep Q-Network system for optimizing push notification scheduling 
using Contextual Markov Decision Process (MDP).
"""

from .env import NotificationEnv, UserProfile
from .networks import QNetwork, get_device
from .agents import DQNAgent, ReplayBuffer
from .utils import (
    compute_moving_average,
    plot_learning_curve,
    plot_behavioral_heatmap,
    MetricsLogger
)

__all__ = [
    # Environment
    "NotificationEnv",
    "UserProfile",
    # Networks
    "QNetwork",
    "get_device",
    # Agents
    "DQNAgent",
    "ReplayBuffer",
    # Utils
    "compute_moving_average",
    "plot_learning_curve",
    "plot_behavioral_heatmap",
    "MetricsLogger",
]

__version__ = "1.0.0"
