"""
env.py - The User Simulation (Contextual MDP)

This module implements a Gymnasium environment that simulates user behavior
for push notification scheduling. The environment models users with different
personas (e.g., students, workers) and their varying responsiveness patterns
throughout the day.

Mathematical Framework:
-----------------------
State Space (S): A continuous vector of shape (5,)
    [norm_hour, norm_day, norm_recency, norm_annoyance, user_id_encoded]
    
Action Space (A): Discrete(2)
    {0: Wait, 1: Send}
    
Transition Logic (P):
    - Time advances by 1 hour every step
    - Click probability depends on circadian rhythm and context
    
Reward Function (R):
    - Wait: 0
    - Send + Click: +10.0
    - Send + Ignore: -1.0
    - Churn: -50.0
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class UserProfile:
    """
    Represents a user persona with behavioral characteristics.
    
    Attributes:
        user_id: Unique identifier for the persona (used in state encoding)
        wake_hour: Hour when the user typically wakes up (0-23)
        sleep_hour: Hour when the user typically goes to sleep (0-23)
        work_hours: List of hours during which the user is working
        responsiveness: Base probability of clicking a notification (0.0-1.0)
        patience: Decay rate for annoyance per step (0.0-1.0)
    
    Example Profiles:
        Student (Night Owl): wake_hour=12, sleep_hour=4, work_hours=[]
        Worker (9-5 Job): wake_hour=7, sleep_hour=23, work_hours=[9,10,...,17]
    """
    user_id: int
    wake_hour: int = 7
    sleep_hour: int = 23
    work_hours: List[int] = field(default_factory=lambda: list(range(9, 18)))
    responsiveness: float = 0.5
    patience: float = 0.9
    
    def __post_init__(self):
        """Validate profile parameters."""
        assert 0 <= self.wake_hour <= 23, "wake_hour must be in [0, 23]"
        assert 0 <= self.sleep_hour <= 23, "sleep_hour must be in [0, 23]"
        assert 0.0 <= self.responsiveness <= 1.0, "responsiveness must be in [0, 1]"
        assert 0.0 <= self.patience <= 1.0, "patience must be in [0, 1]"
        
    def is_awake(self, hour: int) -> bool:
        """
        Determine if the user is awake at a given hour.
        
        Handles wrap-around for night owls (e.g., sleep at 4am, wake at 12pm).
        """
        hour = hour % 24
        if self.wake_hour < self.sleep_hour:
            # Normal schedule: wake at 7, sleep at 23
            return self.wake_hour <= hour < self.sleep_hour
        else:
            # Night owl schedule: wake at 12, sleep at 4
            # Awake from wake_hour to midnight OR from midnight to sleep_hour
            return hour >= self.wake_hour or hour < self.sleep_hour
            
    def is_working(self, hour: int) -> bool:
        """Check if the user is working at a given hour."""
        return (hour % 24) in self.work_hours


class NotificationEnv(gym.Env):
    """
    Gymnasium environment for push notification scheduling.
    
    The environment simulates a user receiving push notifications over time.
    The agent must learn when to send notifications to maximize engagement
    while avoiding user annoyance and churn.
    
    Observation Space:
        Box(4 + num_users,) with components:
        - norm_hour: Current hour normalized to [0, 1]
        - norm_day: Current day normalized to [0, 1] (7-day cycle)
        - norm_recency: Hours since last notification sent, normalized [0, 1]
        - norm_annoyance: Current annoyance level normalized [0, 1]
        - user_one_hot: One-hot encoded user ID (categorical representation)
        
    Action Space:
        Discrete(2):
        - 0: Wait (do nothing)
        - 1: Send notification
        
    Reward Structure:
        - Wait: 0
        - Send + User clicks: +10.0 (default)
        - Send + User ignores: -3.0 (default - increased for scientific rigor)
        - Churn (annoyance > threshold): -50.0 (terminal)
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    # Constants for normalization
    MAX_RECENCY = 48.0  # Maximum hours to track since last send
    MAX_ANNOYANCE = 10.0  # Maximum annoyance for normalization
    CHURN_THRESHOLD = 3.0  # Reduced from 5.0 for higher sensitivity
    MAX_EPISODE_STEPS = 168  # One week in hours
    
    def __init__(
        self,
        user_profile: UserProfile,
        num_users: int = 2,
        max_episode_steps: int = 168,
        reward_click: float = 10.0,
        reward_ignore: float = -3.0,
        reward_wait: float = 0.0,
        reward_churn: float = -50.0,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the notification environment.
        
        Args:
            user_profile: UserProfile instance defining user behavior
            num_users: Total number of unique user personas for one-hot encoding
            max_episode_steps: Maximum steps per episode (default: 168 = 1 week)
            render_mode: Rendering mode ("human" or "ansi")
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.user_profile = user_profile
        self.num_users = num_users
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        
        # Reward configuration
        self.reward_click = reward_click
        self.reward_ignore = reward_ignore
        self.reward_wait = reward_wait
        self.reward_churn = reward_churn
        
        # Define observation and action spaces
        # 4 continuous features + OHE vector for users
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4 + self.num_users,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(2)
        
        # Initialize state variables
        self._current_hour: int = 0
        self._current_day: int = 0
        self._hours_since_send: int = 24
        self._annoyance: float = 0.0
        self._step_count: int = 0
        
        # Set random seed
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            self.np_random = np.random.default_rng()
            
    def _get_state(self) -> np.ndarray:
        """
        Construct the normalized state vector with One-Hot encoding for User ID.
        """
        norm_hour = self._current_hour / 23.0
        norm_day = self._current_day / 6.0
        norm_recency = min(self._hours_since_send, self.MAX_RECENCY) / self.MAX_RECENCY
        norm_annoyance = min(self._annoyance, self.MAX_ANNOYANCE) / self.MAX_ANNOYANCE
        
        # One-hot encode the user_id
        user_ohe = np.zeros(self.num_users, dtype=np.float32)
        if 0 <= self.user_profile.user_id < self.num_users:
            user_ohe[self.user_profile.user_id] = 1.0
        
        return np.concatenate([
            [norm_hour, norm_day, norm_recency, norm_annoyance],
            user_ohe
        ]).astype(np.float32)
        
    def _compute_click_probability(self) -> float:
        """
        Compute the probability of the user clicking a notification.
        """
        # Base responsiveness
        prob = self.user_profile.responsiveness
        
        # Circadian factor: User must be awake to click
        circadian_factor = 1.0 if self.user_profile.is_awake(self._current_hour) else 0.0
        prob *= circadian_factor
        
        # Work penalty: Less likely to engage during work
        work_penalty = 0.2 if self.user_profile.is_working(self._current_hour) else 1.0
        prob *= work_penalty
        
        # Recency decay: Less likely if sent recently
        recency_decay = 0.1 if self._hours_since_send < 4 else 1.0
        prob *= recency_decay
        
        return prob
        
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
            
        # Reset state variables
        if options and "start_hour" in options:
            self._current_hour = options["start_hour"]
        else:
            self._current_hour = int(self.np_random.integers(0, 24))
            
        # Support dynamic user profile switching at reset
        if options and "user_profile" in options:
            self.user_profile = options["user_profile"]
            
        self._current_day = 0
        self._hours_since_send = 24
        self._annoyance = 0.0
        self._step_count = 0
        
        info = {
            "hour": self._current_hour,
            "day": self._current_day,
            "annoyance": self._annoyance,
            "hours_since_send": self._hours_since_send,
            "user_id": self.user_profile.user_id
        }
        
        return self._get_state(), info
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        self._step_count += 1
        reward = 0.0
        terminated = False
        truncated = False
        clicked = False
        
        if action == 1:  # Send notification
            click_prob = self._compute_click_probability()
            clicked = self.np_random.random() < click_prob
            
            if clicked:
                reward = self.reward_click
                self._annoyance = max(0.0, self._annoyance - 1.0)
            else:
                reward = self.reward_ignore
                self._annoyance += 1.0
                
            self._hours_since_send = 0
        else:  # Wait
            reward = self.reward_wait
            self._hours_since_send += 1
            
        # Apply patience decay to annoyance
        self._annoyance *= self.user_profile.patience
        
        # Check for churn
        if self._annoyance > self.CHURN_THRESHOLD:
            reward = self.reward_churn
            terminated = True
            
        # Advance time
        self._current_hour = (self._current_hour + 1) % 24
        if self._current_hour == 0:
            self._current_day = (self._current_day + 1) % 7
            
        # Check for episode truncation
        if self._step_count >= self.max_episode_steps:
            truncated = True
            
        info = {
            "hour": self._current_hour,
            "day": self._current_day,
            "annoyance": self._annoyance,
            "hours_since_send": self._hours_since_send,
            "clicked": clicked if action == 1 else None,
            "click_probability": self._compute_click_probability(),
            "user_id": self.user_profile.user_id,
            "churned": terminated and self._annoyance > self.CHURN_THRESHOLD
        }
        
        return self._get_state(), reward, terminated, truncated, info
        
    def render(self) -> Optional[str]:
        """Render the current environment state."""
        if self.render_mode == "ansi" or self.render_mode == "human":
            awake_status = "awake" if self.user_profile.is_awake(self._current_hour) else "asleep"
            work_status = "working" if self.user_profile.is_working(self._current_hour) else "free"
            
            output = (
                f"Hour: {self._current_hour:02d}:00 | "
                f"Day: {self._current_day} | "
                f"Status: {awake_status}, {work_status} | "
                f"Annoyance: {self._annoyance:.2f} | "
                f"Hours since send: {self._hours_since_send}"
            )
            
            if self.render_mode == "human":
                print(output)
            return output
            
        return None
        
    def close(self):
        pass


# Predefined user profiles for experiments
STUDENT_PROFILE = UserProfile(
    user_id=0,
    wake_hour=12,
    sleep_hour=4,
    work_hours=[],
    responsiveness=0.6,
    patience=0.85
)

WORKER_PROFILE = UserProfile(
    user_id=1,
    wake_hour=7,
    sleep_hour=23,
    work_hours=list(range(9, 18)),
    responsiveness=0.5,
    patience=0.9
)

WORKAHOLIC_PROFILE = UserProfile(
    user_id=2,
    wake_hour=6,
    sleep_hour=0,
    work_hours=list(range(8, 20)),  # 8am to 8pm
    responsiveness=0.4,
    patience=0.95
)
