"""
networks.py - The Q-Network (PyTorch MLP)

This module implements the neural network architecture for Deep Q-Learning.
The network approximates the Q-function Q(s, a) which estimates the expected
cumulative reward for taking action a in state s.

Architecture:
    Input(D) → Linear(128, ReLU) → Linear(128, ReLU) → Linear(64, ReLU) → Output(2)
    where D = dimension of state vector (4 + num_users).
    
Hardware Optimization:
    Optimized for Apple Silicon (M2) using MPS (Metal Performance Shaders)
    when available, falling back to CPU otherwise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, Tuple


def get_device() -> torch.device:
    """
    Get the optimal compute device for Apple Silicon.
    
    Prioritizes MPS (Metal Performance Shaders) for M1/M2 chips,
    falls back to CPU if MPS is not available.
    
    Returns:
        torch.device: The compute device to use
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class QNetwork(nn.Module):
    """
    Deep Q-Network for notification scheduling.
    
    The network takes a state vector and outputs Q-values for each action.
    Architecture follows best practices for DQN with moderate complexity.
    
    Architecture:
        - Input Layer: D features (state vector)
        - Hidden Layer 1: 128 units with ReLU activation
        - Hidden Layer 2: 128 units with ReLU activation  
        - Hidden Layer 3: 64 units with ReLU activation
        - Output Layer: 2 units (Q-values for Wait/Send actions)
        
    The deeper architecture (3 hidden layers) allows the network to learn
    complex non-linear patterns in user behavior across different contexts.
    """
    
    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 2,
        hidden_dims: tuple = (128, 128, 64),
        device: Union[torch.device, str, None] = None
    ):
        """
        Initialize the Q-Network.
        
        Args:
            state_dim: Dimension of state vector (default: 5)
            action_dim: Number of actions (default: 2)
            hidden_dims: Tuple of hidden layer dimensions
            device: Compute device (default: auto-detect)
        """
        super(QNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Set device
        if device is None:
            self.device = get_device()
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
            
        # Build network layers
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc_out = nn.Linear(hidden_dims[2], action_dim)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
        
        # Move to device
        self.to(self.device)
        
    def _initialize_weights(self):
        """
        Initialize network weights using Xavier uniform initialization.
        
        Xavier initialization helps maintain stable gradients during training,
        which is particularly important for deep networks.
        """
        for module in [self.fc1, self.fc2, self.fc3, self.fc_out]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: State tensor of shape (batch_size, state_dim) or (state_dim,)
            
        Returns:
            Q-values tensor of shape (batch_size, action_dim) or (action_dim,)
        """
        # Ensure input is on the correct device
        if x.device != self.device:
            x = x.to(self.device)
            
        # Handle single state (add batch dimension)
        squeeze_output = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
            
        # Forward pass with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_values = self.fc_out(x)
        
        # Remove batch dimension if added
        if squeeze_output:
            q_values = q_values.squeeze(0)
            
        return q_values
        
    def get_action(self, state: torch.Tensor) -> int:
        """
        Get the greedy action (argmax Q-value) for a given state.
        
        Args:
            state: State tensor
            
        Returns:
            Action index (0 or 1)
        """
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax().item()
            
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values for all actions given a state.
        
        Convenience method for analysis and visualization.
        
        Args:
            state: State tensor
            
        Returns:
            Q-values tensor
        """
        with torch.no_grad():
            return self.forward(state)


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN architecture (optional advanced variant).
    
    Separates the Q-value estimation into:
    - Value stream V(s): How good is this state?
    - Advantage stream A(s, a): How much better is action a than others?
    
    Q(s, a) = V(s) + A(s, a) - mean(A(s, :))
    
    This architecture can provide better learning stability for problems
    where action choice doesn't always matter (e.g., during sleep hours).
    """
    
    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 2,
        hidden_dim: int = 128,
        device: Union[torch.device, str, None] = None
    ):
        """
        Initialize the Dueling Q-Network.
        
        Args:
            state_dim: Dimension of state vector
            action_dim: Number of actions
            hidden_dim: Dimension of hidden layers
            device: Compute device
        """
        super(DuelingQNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        if device is None:
            self.device = get_device()
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
            
        # Shared feature extraction
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        self.to(self.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining value and advantage streams."""
        if x.device != self.device:
            x = x.to(self.device)
            
        squeeze_output = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
            
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine: Q = V + A - mean(A)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        if squeeze_output:
            q_values = q_values.squeeze(0)
            
        return q_values
class DRQNetwork(nn.Module):
    """
    Deep Recurrent Q-Network (DRQN) architecture.
    
    Adds an LSTM layer to the Q-Network to handle Partial Observability (POMDP).
    The network maintains an internal state (hidden and cell states) that 
    allows it to remember past observations and infer latent variables 
    (like user persona without explicit ID).
    """
    
    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 2,
        hidden_dim: int = 128,
        device: Union[torch.device, str, None] = None
    ):
        """
        Initialize the DRQ-Network.
        
        Args:
            state_dim: Dimension of state vector
            action_dim: Number of actions
            hidden_dim: Hidden dimension for LSTM and FC layers
            device: Compute device
        """
        super(DRQNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        if device is None:
            self.device = get_device()
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
            
        # Feature extraction (before LSTM)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        
        # LSTM layer
        # batch_first=True means input shape is (batch, seq_len, features)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Output layer
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        
        self.to(self.device)
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the DRQN.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, state_dim)
            hidden: Initial hidden state (h, c)
            
        Returns:
            q_values: Q-values for each action (batch_size, seq_len, action_dim)
            hidden: Re-calculated hidden state (h, c)
        """
        if x.device != self.device:
            x = x.to(self.device)
            
        # x shape: (batch, seq_len, state_dim)
        batch_size, seq_len, _ = x.size()
        
        # FC1 layer
        x = F.relu(self.fc1(x))
        
        # LSTM layer
        # if hidden is None, LSTM initializes it to zeros
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Output layer
        q_values = self.fc2(lstm_out)
        
        return q_values, hidden

    def init_hidden(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return initial hidden and cell states (zeros)."""
        h = torch.zeros(1, batch_size, self.hidden_dim).to(self.device)
        c = torch.zeros(1, batch_size, self.hidden_dim).to(self.device)
        return (h, c)
