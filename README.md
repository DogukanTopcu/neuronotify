# NeuroNotify: DQN-Based Push Notification Optimization

[![IEEE Conference Paper](https://img.shields.io/badge/Research-IEEE%20Conference-blue.svg)](https://ieee.org)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Hardware: Apple Silicon](https://img.shields.io/badge/Hardware-Apple%20Silicon%20(M2)-green.svg)](https://developer.apple.com/metal/pytorch/)

NeuroNotify is a research project implementing a Deep Q-Network (DQN) to optimize push notification scheduling. By modeling user behavior as a **Contextual Markov Decision Process (MDP)**, the system learns to maximize user engagement (clicks) while minimizing annoyance and churn.

## 🚀 Hardware Optimization
This project is strictly optimized for **Apple Silicon (M1/M2/M3)**. It utilizes the `mps` (Metal Performance Shaders) backend for high-performance neural network training.

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

## 🏗️ Project Architecture

The system is built with a modular structure designed for scientific reproducibility:

```text
neuro_notify/
│
├── src/
│   ├── __init__.py            # Package entry points
│   ├── env.py                 # The User Simulation (Contextual MDP)
│   ├── networks.py            # Q-Network Architecture (PyTorch MLP)
│   ├── agents.py              # DQNAgent and ReplayBuffer classes
│   └── utils.py               # Metrics logging and Visualization helpers
│
├── requirements.txt           # Dependency manifest
├── experiment.ipynb            # Main training and evaluation pipeline
└── README.md                  # Detailed documentation
```

## 🧪 Theoretical Framework

### 1. Contextual MDP
The state space $S$ is a 5-dimensional vector:
- `norm_hour`: Current hour $[0, 1]$
- `norm_day`: Day of week $[0, 1]$
- `norm_recency`: Hours since last notification $[0, 1]$
- `norm_annoyance`: Current user irritation level $[0, 1]$
- `user_id_encoded`: Normalized ID for persona discrimination

### 2. Behavioral Dynamics
The simulation uses a multi-factor click probability model:
$$P(\text{click}) = R \times C \times W \times D$$
Where:
- $R$: Base responsiveness
- $C$: Circadian factor ($1.0$ awake, $0.0$ asleep)
- $W$: Work penalty ($0.2$ during work hours)
- $D$: Recency decay ($0.1$ if sent $< 4$ hours ago)

### 3. Reward Shaping
The policy is guided by a strategic reward function:
- **Send + Click**: $+10.0$
- **Wait**: $0.0$
- **Send + Ignore**: $-1.0$ (interruption cost)
- **Churn**: $-50.0$ (terminal penalty for high annoyance)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd neuronotify
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Running Experiments

Open the `experiment.ipynb` notebook to run the full training and visualization pipeline:

```bash
jupyter notebook experiment.ipynb
```

### Key Visualizations Included:
- **Learning Curves**: Evolution of rewards across 1,000 episodes.
- **Behavioral Heatmaps**: 2D grid (Hour vs. Recency) showing learned Q-values for the "Send" action. These demonstrate the agent's understanding of user sleep/wake cycles.
- **Multi-Persona Comparison**: Visualization showing how the same model adapts to "Student" (Night Owl) vs. "Worker" (9-5) profiles.

## 🎓 Researcher's Summary

The project demonstrates three core RL principles:
1. **Contextual Mapping**: How embedding `user_id` in the state vector allows a single model to handle heterogeneous user groups.
2. **Stability Theory**: The necessity of **Experience Replay** and **Target Networks** to prevent the "Moving Target" problem in discrete action spaces.
3. **Safety Constraints**: Using high terminal penalties to enforce "safe" notification policies that prevent user attrition.

---
*Created for the NeuroNotify Research Paper.*
