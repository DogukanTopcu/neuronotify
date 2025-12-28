"""
agents.py - The Learner (DQN Agent and Replay Buffer)

This module implements the core DQN learning components:
- ReplayBuffer: Experience replay for stable training
- DQNAgent: The learning agent with policy and target networks

Key DQN Concepts:
-----------------
1. Experience Replay: Breaks temporal correlation in training data
2. Target Network: Provides stable Q-value targets for gradient updates
3. Epsilon-Greedy: Balances exploration vs exploitation

Mathematical Foundation:
------------------------
Q-Learning Update Rule:
    Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') - Q(s, a)]
    
DQN Loss Function:
    L = E[(r + γ max_a' Q_target(s', a') - Q(s, a))²]
"""

import random
from collections import deque
from typing import Tuple, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import QNetwork, get_device


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.
    
    Stores transitions (s, a, r, s', done) and enables random sampling
    for training. This breaks the temporal correlation between consecutive
    samples, which is crucial for stable neural network training.
    
    The buffer implements a FIFO queue with a maximum capacity. When full,
    the oldest experiences are discarded to make room for new ones.
    
    Attributes:
        capacity: Maximum number of transitions to store
        buffer: Deque containing the transitions
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store (default: 10000)
        """
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state vector
            action: Action taken (0 or 1)
            reward: Reward received
            next_state: Resulting state vector
            done: Whether the episode terminated
        """
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)
        
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of numpy arrays: (states, actions, rewards, next_states, dones)
        """
        transitions = random.sample(self.buffer, batch_size)
        
        # Unzip the transitions
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
        
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
        
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= batch_size


class DQNAgent:
    """
    Deep Q-Network Agent for notification scheduling.
    
    The agent maintains two networks:
    - policy_net: The network being actively trained
    - target_net: A periodically updated copy for stable Q-target computation
    
    The separation of networks addresses the "moving target" problem where
    bootstrap targets change during training, causing instability.
    
    Hyperparameters:
        - gamma (γ): Discount factor for future rewards
        - epsilon: Exploration rate for ε-greedy policy
        - learning_rate: Step size for gradient descent
        - batch_size: Number of samples per training step
        
    Training Process:
        1. Collect experience using ε-greedy policy
        2. Store transitions in replay buffer
        3. Sample random minibatch from buffer
        4. Compute Q-targets using target network
        5. Update policy network via gradient descent
        6. Periodically sync target network
    """
    
    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 2,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        use_double_dqn: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the DQN Agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            learning_rate: Learning rate for Adam optimizer
            gamma: Discount factor (0-1)
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Multiplicative decay factor per episode
            buffer_capacity: Replay buffer size
            batch_size: Training batch size
            use_double_dqn: Whether to use Double DQN update rule (default: True)
            device: Compute device (auto-detect if None)
        """
        # Set device (Apple Silicon optimized)
        self.device = device if device else get_device()
        
        # Network dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_double_dqn = use_double_dqn
        
        # Initialize networks
        self.policy_net = QNetwork(state_dim, action_dim, device=self.device)
        self.target_net = QNetwork(state_dim, action_dim, device=self.device)
        
        # Copy policy weights to target
        self.update_target()
        
        # Target network should not be trained directly
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Training statistics
        self.training_steps = 0
        self.loss_history: List[float] = []

    def act(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """
        Select action using epsilon-greedy policy.
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        # Exploration: random action
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
            
        # Exploitation: greedy action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            # Ensure it has batch dimension for forward pass
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
            
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        
    def train_step(self, batch_size: Optional[int] = None) -> Optional[float]:
        """
        Perform one training step using experience replay.
        
        Implements Standard DQN or Double DQN (DDQN) based on self.use_double_dqn.
        DDQN mitigates overestimation bias by using the policy network to select
        the action and the target network to evaluate it.
        """
        batch_size = batch_size or self.batch_size
        
        # Check if enough samples available
        if not self.replay_buffer.is_ready(batch_size):
            return None
            
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values: Q(s, a)
        q_values = self.policy_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute Q-targets
        with torch.no_grad():
            if self.use_double_dqn:
                # DDQN: Use policy_net to choose action, target_net to evaluate it
                next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN: Use target_net for both selection and evaluation
                next_q_values = self.target_net(next_states).max(1)[0]
            
            # Target: r + γ * Q(s', a') * (1 - done)
            q_targets = rewards + self.gamma * next_q_values * (1 - dones)
            
        # Compute loss
        loss = self.loss_fn(q_values, q_targets)
        
        # Gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Track statistics
        self.training_steps += 1
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        return loss_value
        
    def update_target(self) -> None:
        """
        Update target network with policy network weights.
        
        This should be called periodically (e.g., every N episodes)
        to provide stable Q-targets while allowing the policy to improve.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def decay_epsilon(self) -> None:
        """
        Decay exploration rate.
        
        Multiplies epsilon by decay factor, with a floor at epsilon_end.
        This implements a gradual shift from exploration to exploitation.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions given a state.
        
        Useful for analysis and visualization.
        
        Args:
            state: State vector
            
        Returns:
            Array of Q-values [Q(s, Wait), Q(s, Send)]
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.cpu().numpy()
            
    def save(self, filepath: str) -> None:
        """
        Save agent state to file.
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'loss_history': self.loss_history
        }
        torch.save(checkpoint, filepath)
        
    def load(self, filepath: str) -> None:
        """
        Load agent state from file.
        
        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        self.loss_history = checkpoint.get('loss_history', [])
