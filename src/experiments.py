"""
experiments.py - Standardized Training Loops for NeuroNotify
"""

import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple

from src.env_stochastic import StochasticNotificationEnv
from src.agents import DQNAgent, DRQNAgent
from src.utils import evaluate_agent, MetricsLogger, get_device

def run_standardized_experiment(
    user_profiles: List[Any],
    num_episodes: int = 3000,
    include_user_id: bool = True,
    learning_rate: float = 0.0005,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.999,
    target_update_freq: int = 10,
    eval_every: int = 100,
    seed: int = 42
) -> Tuple[DQNAgent, MetricsLogger, List[float]]:
    """
    Run a standardized training loop for Experiment 3.
    
    This function ensures the correct state dimensions are used and 
    handles the training/evaluation cycle consistently.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Initialize env to get state dimensions
    temp_env = StochasticNotificationEnv(
        user_profile=user_profiles[0],
        num_users=len(user_profiles),
        include_user_id=include_user_id,
        seed=seed
    )
    
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.n
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        use_double_dqn=True
    )
    
    logger = MetricsLogger()
    eval_rewards = []
    
    for episode in tqdm(range(num_episodes), desc=f"Training (OHE={include_user_id})"):
        # Sample random profile per episode
        profile = np.random.choice(user_profiles)
        
        env = StochasticNotificationEnv(
            user_profile=profile,
            num_users=len(user_profiles),
            include_user_id=include_user_id
        )
        
        state, info = env.reset()
        episode_reward = 0
        done = False
        clicks = 0
        sends = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, terminated)
            
            loss = agent.train_step()
            if loss:
                logger.log_training_step(loss, agent.epsilon)
                
            state = next_state
            episode_reward += reward
            done = terminated or truncated
            
            if action == 1:
                sends += 1
                if info.get('clicked'):
                    clicks += 1
        
        agent.decay_epsilon()
        
        if episode % target_update_freq == 0:
            agent.update_target()
            
        logger.log_episode(
            reward=episode_reward,
            length=env._step_count,
            clicks=clicks,
            sends=sends,
            churned=info.get('churned', False)
        )
        
        # Periodic Evaluation
        if (episode + 1) % eval_every == 0:
            avg_eval_reward = evaluate_agent(
                agent, 
                temp_env, 
                n_episodes=10, 
                user_profiles=user_profiles
            )
            eval_rewards.append(avg_eval_reward)
            
    return agent, logger, eval_rewards

def run_drqn_experiment(
    user_profiles: List[Any],
    num_episodes: int = 2000,
    include_user_id: bool = False,
    hidden_dim: int = 128,
    learning_rate: float = 0.0005,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.999,
    seq_len: int = 16,
    batch_size: int = 32,
    target_update_freq: int = 10,
    eval_every: int = 100,
    seed: int = 42
) -> Tuple[DRQNAgent, MetricsLogger, List[float]]:
    """
    Run a standardized training loop for DRQN (Recurrent Agent).
    
    Handles episode collection and sequence-based training required for LSTMs.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Initialize env
    temp_env = StochasticNotificationEnv(
        user_profile=user_profiles[0],
        num_users=len(user_profiles),
        include_user_id=include_user_id,
        seed=seed
    )
    
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.n
    
    agent = DRQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        seq_len=seq_len,
        batch_size=batch_size
    )
    
    logger = MetricsLogger()
    eval_rewards = []
    
    for episode in tqdm(range(num_episodes), desc=f"DRQN Training (OHE={include_user_id})"):
        profile = np.random.choice(user_profiles)
        
        env = StochasticNotificationEnv(
            user_profile=profile,
            num_users=len(user_profiles),
            include_user_id=include_user_id
        )
        
        state, info = env.reset()
        agent.reset_hidden()
        
        episode_data = []
        episode_reward = 0
        done = False
        clicks = 0
        sends = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Store transition for later episode processing
            episode_data.append((state, action, reward, next_state, terminated))
            
            state = next_state
            episode_reward += reward
            done = terminated or truncated
            
            if action == 1:
                sends += 1
                if info.get('clicked'):
                    clicks += 1
        
        # Store full episode and train
        agent.store_episode(episode_data)
        
        # Perform training steps (one per episode)
        loss = agent.train_step()
        if loss:
            logger.log_training_step(loss, agent.epsilon)
            
        agent.decay_epsilon()
        
        if episode % target_update_freq == 0:
            agent.update_target()
            
        logger.log_episode(
            reward=episode_reward,
            length=env._step_count,
            clicks=clicks,
            sends=sends,
            churned=info.get('churned', False)
        )
        
        # Periodic Evaluation
        if (episode + 1) % eval_every == 0:
            avg_eval_reward = evaluate_agent(
                agent, 
                temp_env, 
                n_episodes=10, 
                user_profiles=user_profiles
            )
            eval_rewards.append(avg_eval_reward)
            
    return agent, logger, eval_rewards
