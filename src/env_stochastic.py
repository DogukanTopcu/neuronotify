"""
env_stochastic.py - Stochastic User Simulation for Experiment 3

This module extends the base NotificationEnv to introduce real-world variance:
1. Stochastic wake/sleep times (with configurable noise)
2. Optional removal of User ID OHE (for POMDP / Persona Inference)
3. Support for comparing RL against heuristic baselines

Mathematical Framework:
-----------------------
State Space (S): 
    - With OHE: [norm_hour, norm_day, norm_recency, norm_annoyance, user_ohe...]
    - Without OHE (POMDP): [norm_hour, norm_day, norm_recency, norm_annoyance, is_working, is_awake]
    
The key difference: Without OHE, the agent must infer the persona from behavioral features.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.env import UserProfile, NotificationEnv


@dataclass
class StochasticUserProfile(UserProfile):
    """
    User profile with stochastic schedule variance.
    
    Extends UserProfile to add noise to wake/sleep times, simulating
    real-world irregularity in user schedules.
    
    Attributes:
        schedule_noise_std: Standard deviation (in hours) for wake/sleep time variance
    """
    schedule_noise_std: float = 1.0  # hours
    
    def __post_init__(self):
        super().__post_init__()
        assert self.schedule_noise_std >= 0.0, "schedule_noise_std must be non-negative"
        
    def sample_wake_hour(self, rng: np.random.Generator) -> int:
        """Sample a wake hour with Gaussian noise."""
        noisy_hour = self.wake_hour + rng.normal(0, self.schedule_noise_std)
        return int(np.clip(noisy_hour, 0, 23))
        
    def sample_sleep_hour(self, rng: np.random.Generator) -> int:
        """Sample a sleep hour with Gaussian noise."""
        noisy_hour = self.sleep_hour + rng.normal(0, self.schedule_noise_std)
        return int(np.clip(noisy_hour, 0, 23))


class StochasticNotificationEnv(NotificationEnv):
    """
    Enhanced environment with stochastic schedules and optional POMDP mode.
    
    Key Features:
    - Stochastic wake/sleep times (configurable variance)
    - Optional removal of User ID from observation (POMDP mode)
    - Compatible with existing DQN agent
    
    POMDP Mode:
        When `include_user_id=False`, the observation space excludes OHE.
        The agent must rely on behavioral features (is_working, is_awake)
        to infer the persona and generalize its policy.
    """
    
    def __init__(
        self,
        user_profile: StochasticUserProfile,
        num_users: int = 3,
        max_episode_steps: int = 168,
        reward_click: float = 10.0,
        reward_ignore: float = -3.0,
        reward_wait: float = 0.0,
        reward_churn: float = -15.0,
        include_user_id: bool = True,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the stochastic notification environment.
        
        Args:
            user_profile: StochasticUserProfile instance
            num_users: Total number of unique user personas
            max_episode_steps: Maximum steps per episode
            reward_click: Reward for user engagement
            reward_ignore: Penalty for ignored notification
            reward_wait: Reward for waiting
            reward_churn: Penalty for user churn
            include_user_id: If False, removes OHE from observation (POMDP mode)
            render_mode: Rendering mode
            seed: Random seed
        """
        self.include_user_id = include_user_id
        
        # Initialize parent (will set observation_space)
        super().__init__(
            user_profile=user_profile,
            num_users=num_users,
            max_episode_steps=max_episode_steps,
            reward_click=reward_click,
            reward_ignore=reward_ignore,
            reward_wait=reward_wait,
            reward_churn=reward_churn,
            render_mode=render_mode,
            seed=seed
        )
        
        # Override observation space to include behavioral features (is_working, is_awake)
        if self.include_user_id:
            # State: [norm_hour, norm_day, norm_recency, norm_annoyance, is_working, is_awake, user_ohe...]
            # Shape: 4 (base) + 2 (behavioral) + num_users (OHE)
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(6 + self.num_users,), 
                dtype=np.float32
            )
        else:
            # State: [norm_hour, norm_day, norm_recency, norm_annoyance, is_working, is_awake]
            # Shape: 4 (base) + 2 (behavioral)
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(6,), 
                dtype=np.float32
            )
        
        # Track episode-specific sampled schedule
        self._episode_wake_hour: int = user_profile.wake_hour
        self._episode_sleep_hour: int = user_profile.sleep_hour
        self._consecutive_ignores: int = 0
        
    def _get_state(self) -> np.ndarray:
        """
        Construct the state vector.
        
        Returns:
            - If include_user_id=True: [hour, day, recency, annoyance, user_ohe]
            - If include_user_id=False: [hour, day, recency, annoyance, is_working, is_awake]
        """
        norm_hour = self._current_hour / 23.0
        norm_day = self._current_day / 6.0
        norm_recency = min(self._hours_since_send, self.MAX_RECENCY) / self.MAX_RECENCY
        norm_annoyance = min(self._annoyance, self.MAX_ANNOYANCE) / self.MAX_ANNOYANCE
        
        base_features = [norm_hour, norm_day, norm_recency, norm_annoyance]
        
        # Always include behavioral features to ensure OHE has at least as much info as POMDP
        is_working = 1.0 if self.user_profile.is_working(self._current_hour) else 0.0
        is_awake = 1.0 if self._is_awake_current_episode(self._current_hour) else 0.0
        context_features = base_features + [is_working, is_awake]
        
        if self.include_user_id:
            # Standard mode: include OHE + context
            user_ohe = np.zeros(self.num_users, dtype=np.float32)
            if 0 <= self.user_profile.user_id < self.num_users:
                user_ohe[self.user_profile.user_id] = 1.0
            return np.concatenate([context_features, user_ohe]).astype(np.float32)
        else:
            # POMDP mode: just context
            return np.array(context_features, dtype=np.float32)
    
    def _is_awake_current_episode(self, hour: int) -> bool:
        """
        Check if user is awake based on episode-specific sampled schedule.
        """
        hour = hour % 24
        if self._episode_wake_hour < self._episode_sleep_hour:
            return self._episode_wake_hour <= hour < self._episode_sleep_hour
        else:
            return hour >= self._episode_wake_hour or hour < self._episode_sleep_hour
    
    def _compute_click_probability(self) -> float:
        """
        Compute click probability using episode-specific schedule.
        """
        prob = self.user_profile.responsiveness
        
        # Use episode-specific wake/sleep schedule
        circadian_factor = 1.0 if self._is_awake_current_episode(self._current_hour) else 0.0
        prob *= circadian_factor
        
        work_penalty = 0.2 if self.user_profile.is_working(self._current_hour) else 1.0
        prob *= work_penalty
        
        recency_decay = 0.1 if self._hours_since_send < 4 else 1.0
        prob *= recency_decay
        
        return prob
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step with non-linear strike-based rewards.
        """
        # Call parent generic step
        state, reward, terminated, truncated, info = super().step(action)
        
        # Implement Strike System (Non-Linear Penalty)
        if action == 1:
            if not info.get('clicked', False):
                self._consecutive_ignores += 1
                # Strikes escalate penalties more than simple linear annoyance
                if self._consecutive_ignores >= 3:
                    reward = self.reward_churn
                    terminated = True
                    info['churned'] = True
                    info['strike_out'] = True
            else:
                self._consecutive_ignores = 0
        
        # Enhanced info for Oracle and visualization
        info['is_awake'] = self._is_awake_current_episode(self._current_hour)
        info['is_working'] = self.user_profile.is_working(self._current_hour)
        info['consecutive_ignores'] = self._consecutive_ignores
        
        return self._get_state(), reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset with stochastic schedule sampling.
        """
        # Reset parent state
        state, info = super().reset(seed=seed, options=options)
        
        # Sample episode-specific schedule with noise
        if isinstance(self.user_profile, StochasticUserProfile):
            self._episode_wake_hour = self.user_profile.sample_wake_hour(self.np_random)
            self._episode_sleep_hour = self.user_profile.sample_sleep_hour(self.np_random)
        else:
            self._episode_wake_hour = self.user_profile.wake_hour
            self._episode_sleep_hour = self.user_profile.sleep_hour
        
        # Add schedule info to returned info dict
        info['episode_wake_hour'] = self._episode_wake_hour
        info['episode_sleep_hour'] = self._episode_sleep_hour
        self._consecutive_ignores = 0
        
        # Regenerate state with correct observation space
        return self._get_state(), info


# Predefined stochastic profiles
STOCHASTIC_STUDENT_PROFILE = StochasticUserProfile(
    user_id=0,
    wake_hour=12,
    sleep_hour=4,
    work_hours=[],
    responsiveness=0.6,
    patience=0.85,
    schedule_noise_std=1.5  # High variance (night owl irregularity)
)

STOCHASTIC_WORKER_PROFILE = StochasticUserProfile(
    user_id=1,
    wake_hour=7,
    sleep_hour=23,
    work_hours=list(range(9, 18)),
    responsiveness=0.5,
    patience=0.9,
    schedule_noise_std=0.8  # Moderate variance (consistent routine)
)

STOCHASTIC_WORKAHOLIC_PROFILE = StochasticUserProfile(
    user_id=2,
    wake_hour=6,
    sleep_hour=0,
    work_hours=list(range(8, 20)),
    responsiveness=0.4,
    patience=0.95,
    schedule_noise_std=1.2  # Moderate-high variance (unpredictable)
)
