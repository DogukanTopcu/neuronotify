"""
baselines.py - Heuristic Baseline Agents

This module implements non-learning baseline agents for comparison
against the RL agent. These baselines help quantify the "Intelligence Premium"
of the reinforcement learning approach.

Baselines:
    1. RandomAgent: Sends notifications with a fixed probability
    2. EveningOnlyAgent: Static schedule (sends only during evening hours)
    3. ActivityTriggeredAgent: Sends when user is awake and not working
"""

from typing import Optional
import numpy as np


class BaselineAgent:
    """Base class for heuristic agents."""
    
    def act(self, state: np.ndarray, info: Optional[dict] = None) -> int:
        """
        Select an action based on the current state.
        
        Args:
            state: Current observation from environment
            info: Optional info dict from environment (may contain hour, user status, etc.)
            
        Returns:
            Action (0 = Wait, 1 = Send)
        """
        raise NotImplementedError


class RandomAgent(BaselineAgent):
    """
    Random baseline: sends notifications with fixed probability.
    
    This represents the simplest possible strategy and serves as
    a lower bound for performance comparison.
    """
    
    def __init__(self, send_probability: float = 0.1, seed: Optional[int] = None):
        """
        Initialize random agent.
        
        Args:
            send_probability: Probability of sending at each step
            seed: Random seed for reproducibility
        """
        self.send_probability = send_probability
        self.rng = np.random.default_rng(seed)
        
    def act(self, state: np.ndarray, info: Optional[dict] = None) -> int:
        """Randomly decide whether to send."""
        return 1 if self.rng.random() < self.send_probability else 0


class EveningOnlyAgent(BaselineAgent):
    """
    Static schedule baseline: sends only during evening hours (18:00-22:00).
    
    This represents a simple rule-based strategy that many marketing
    teams might use in practice.
    """
    
    def __init__(self, evening_start: int = 18, evening_end: int = 22):
        """
        Initialize evening-only agent.
        
        Args:
            evening_start: Start hour for evening window (inclusive)
            evening_end: End hour for evening window (exclusive)
        """
        self.evening_start = evening_start
        self.evening_end = evening_end
        
    def act(self, state: np.ndarray, info: Optional[dict] = None) -> int:
        """Send if current hour is in evening window."""
        if info and 'hour' in info:
            hour = info['hour']
            if self.evening_start <= hour < self.evening_end:
                return 1
        return 0


class ActivityTriggeredAgent(BaselineAgent):
    """
    Activity-triggered baseline: sends when user is awake and not working.
    
    This represents a more sophisticated heuristic that considers
    user context. It approximates what a "smart" rule-based system
    might do without learning.
    """
    
    def __init__(self, min_recency_hours: int = 12, seed: Optional[int] = None):
        """
        Initialize activity-triggered agent.
        
        Args:
            min_recency_hours: Minimum hours between notifications
            seed: Random seed for tie-breaking
        """
        self.min_recency_hours = min_recency_hours
        self.rng = np.random.default_rng(seed)
        
    def act(self, state: np.ndarray, info: Optional[dict] = None) -> int:
        """
        Send if:
        1. User is awake (from state or info)
        2. User is not working (from info)
        3. Sufficient time has passed since last send
        """
        if info is None:
            # Fall back to random with low probability
            return 1 if self.rng.random() < 0.05 else 0
        
        # Check recency constraint
        hours_since_send = info.get('hours_since_send', 0)
        if hours_since_send < self.min_recency_hours:
            return 0
        
        # POMDP mode: state might include is_awake/is_working directly
        if len(state) == 6:  # POMDP format: [..., is_working, is_awake]
            is_working = state[4] > 0.5
            is_awake = state[5] > 0.5
        else:
            # Standard mode: use info dict
            is_awake = True  # Default assumption if not available
            is_working = False
            
            # Try to extract from info
            if 'hour' in info:
                # We don't have direct access to user profile here,
                # so we use a heuristic: working hours are 9-17
                hour = info['hour']
                is_working = 9 <= hour < 17
        
        # Send if awake and not working
        if is_awake and not is_working:
            return 1
        return 0


class OptimalStaticAgent(BaselineAgent):
    """
    Optimal static schedule baseline.
    
    This agent sends notifications at pre-computed optimal times
    based on average click probabilities across all personas.
    It represents the best possible fixed schedule.
    """
    
    def __init__(self, optimal_hours: list = None):
        """
        Initialize optimal static agent.
        
        Args:
            optimal_hours: List of hours when to send (default: [19, 20, 21])
        """
        self.optimal_hours = optimal_hours if optimal_hours else [19, 20, 21]
        
    def act(self, state: np.ndarray, info: Optional[dict] = None) -> int:
        """Send if current hour is in optimal set."""
        if info and 'hour' in info:
            hour = info['hour']
            if hour in self.optimal_hours:
                return 1
        return 0
