import torch
import numpy as np
from src.env import NotificationEnv, STUDENT_PROFILE
from src.agents import DQNAgent

def test_rigor():
    print("Testing One-Hot Encoding and DDQN forward pass...")
    
    # 1. Test Env OHE
    num_users = 3
    env = NotificationEnv(user_profile=STUDENT_PROFILE, num_users=num_users)
    state, info = env.reset()
    
    # State should be 4 + 3 = 7
    print(f"State shape: {state.shape}")
    assert state.shape[0] == 4 + num_users
    
    # Last 3 elements should be [1, 0, 0] since STUDENT_PROFILE has user_id=0
    print(f"Last 3 elements (OHE): {state[-3:]}")
    assert np.all(state[-3:] == [1.0, 0.0, 0.0])
    
    # 2. Test Agent with 7 dims
    agent = DQNAgent(state_dim=7, action_dim=2, use_double_dqn=True)
    action = agent.act(state)
    print(f"Agent Action: {action}")
    
    # 3. Test Step
    next_state, reward, term, trunc, info = env.step(action)
    print(f"Step Reward: {reward}")
    print(f"Next State shape: {next_state.shape}")
    assert next_state.shape[0] == 7
    
    print("✅ Rigor Verification Successful!")

if __name__ == "__main__":
    test_rigor()
