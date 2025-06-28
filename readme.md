# **Andromeda2: Hierarchical RL for Rocket League**

A high-performance Rocket League agent implementing Hierarchical Reinforcement Learning (HRL) with explicit separation between strategic planning ("brain") and mechanical control ("muscles").

## **🏆 Overview**

Andromeda2 moves beyond purely reactive control by implementing a genuine capacity for strategic planning and long-term decision-making. The agent uses the **official Extended Long Short-Term Memory (xLSTM)** implementation for strategic planning and a fast **Multi-Layer Perceptron (MLP)** for real-time action execution.

### **Key Features**
- **Hierarchical Architecture**: Strategic planner (brain) + Motor controller (muscles)
- **Dual Reward System**: Extrinsic rewards for planner, intrinsic rewards for controller
- **xLSTM Planning**: Advanced memory structures for temporal reasoning
- **Vectorized Training**: High-performance parallel environment execution
- **Goal Vector Interface**: 12D physical state representation for strategic communication

## **Architecture**

### **Strategic Planner (The "Brain")**
- **Architecture**: Official xLSTM implementation with sLSTM and mLSTM blocks
- **Function**: Processes game history to understand strategic landscape  
- **Output**: 12-dimensional goal vector representing strategic intent
- **Reward**: Extrinsic rewards from game outcomes (goals, saves, demos)
- **Features**: Exponential gating, matrix memory, advanced normalization

### **Motor Controller (The "Muscles")**
- **Architecture**: Fast Multi-Layer Perceptron (MLP)
- **Function**: Real-time action execution based on current state + goal vector
- **Input**: Game state + goal vector from planner
- **Output**: Continuous control actions (throttle, steer, pitch, yaw, roll, etc.)
- **Reward**: Intrinsic rewards for achieving goal vector targets

### **Goal Vector Specification**
The 12-dimensional goal vector encodes desired physical state:
- **Target Car Velocity (3D)**: [car_vel_x, car_vel_y, car_vel_z]
- **Target Ball Velocity (3D)**: [ball_vel_x, ball_vel_y, ball_vel_z]
- **Target Car-to-Ball Position (3D)**: [car_to_ball_x, y, z]
- **Target Ball-to-Goal Position (3D)**: [ball_to_goal_x, y, z]

## **Installation**

### **Prerequisites**
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM (for vectorized training)

### **Setup**
```bash
# Clone repository
git clone https://github.com/your-username/Andromeda2.git
cd Andromeda2

# Install dependencies
pip install -r requirements.txt

# Install official xLSTM library
pip install git+https://github.com/NX-AI/xlstm.git

# Install RLGym dependencies
pip install rlgym-sim rocket-league-gym

# Optional: Install development dependencies
pip install -e .
```

### **Additional Setup**
For RLGym-sim, you may need to install Rocket League or RocketSim:
```bash
# Follow RLGym-sim documentation for environment setup
# https://github.com/AechPro/rocket-league-gym-sim
```

## **Training**

### **Quick Start**
```bash
# Train with default configuration
python train.py

# Train with custom config
python train.py --config configs/your_config.yaml

# Resume from checkpoint
python train.py --resume checkpoints/model_checkpoint.pt

# Train with custom run name
python train.py --run-name "experiment_1"
```

### **Training Configuration**
Modify `configs/training_config.yaml` to customize:
- Environment settings (1v1, 2v2, 3v3)
- Agent architecture parameters
- Training hyperparameters
- Logging and evaluation settings

### **Key Training Parameters**
```yaml
# Environment
environment:
  type: "1v1"           # Environment type
  num_envs: 16          # Parallel environments
  hierarchical: true    # Use hierarchical wrapper

# Agent
agent:
  goal_vector_dim: 12   # Goal vector dimension
  planner_update_freq: 8 # Planner update frequency
  training_mode: "hierarchical" # Training mode

# Training
training:
  total_timesteps: 10000000  # Total training steps
  learning_rate: 3e-4        # Learning rate
  batch_size: 64             # Batch size
  n_steps: 2048             # Rollout length
```

### **Monitoring Training**
- **TensorBoard**: `tensorboard --logdir ./logs`
- **Weights & Biases**: Configure in `training_config.yaml`
- **Console Output**: Real-time training metrics

## **Evaluation**

### **Basic Evaluation**
```bash
# Evaluate trained model
python evaluate.py checkpoints/model_final.pt

# Evaluate with specific settings
python evaluate.py checkpoints/model_final.pt \
  --episodes 100 \
  --env-type 1v1 \
  --deterministic

# Compare with baseline
python evaluate.py checkpoints/model_final.pt \
  --baseline baseline_results.npz
```

### **Evaluation Outputs**
- Performance metrics (win rate, goals, episode length)
- Goal vector analysis and visualization
- Detailed episode statistics
- Comparison with baseline models

## **Project Structure**

```
Andromeda2/
├── src/
│   ├── models/
│   │   ├── agent.py         # Main hierarchical agent
│   │   ├── planner.py       # xLSTM strategic planner
│   │   └── controller.py    # MLP motor controller
│   ├── training/
│   │   └── ppo_hierarchical.py # Hierarchical PPO trainer
│   ├── environments/
│   │   ├── factory.py       # Environment factory
│   │   └── vectorized.py    # Vectorized environments
│   └── utils/
│       ├── memory.py        # Rollout buffer
│       └── metrics.py       # Training metrics
├── configs/
│   └── training_config.yaml # Training configuration
├── train.py                 # Main training script
├── evaluate.py             # Evaluation script
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## **Configuration**

### **Environment Types**
- `1v1`: Standard 1v1 matches
- `2v2`: 2v2 team matches
- `3v3`: Full 3v3 matches
- `training`: Specialized training scenarios

### **Training Modes**
- `hierarchical`: Full hierarchical training (default)
- `planner_only`: Train only the strategic planner
- `controller_only`: Train only the motor controller

### **Controller Types**
- `basic`: Standard MLP controller
- `adaptive`: Controller with performance-based adaptation
- `ensemble`: Ensemble of multiple controllers

## **Advanced Features**

### **Curriculum Learning**
```yaml
experimental:
  use_curriculum: true
  curriculum_stages:
    - stage: "basic"
      episodes: 1000000
    - stage: "advanced"
      episodes: 2000000
```

### **Self-Play Training**
```yaml
experimental:
  use_self_play: true
  self_play_frequency: 1000
```

### **Goal Vector Analysis**
Monitor and analyze goal vector evolution:
```python
# Analyze goal vectors during evaluation
python evaluate.py model.pt --episodes 50
# Generates goal vector analysis plots and statistics
```

## **Performance Monitoring**

### **Key Metrics**
- **Planner Performance**: Extrinsic rewards, value function accuracy
- **Controller Performance**: Intrinsic rewards, goal achievement
- **Overall Performance**: Win rate, goals scored/conceded, episode length
- **Goal Vector Analysis**: Stability, component usage, evolution patterns

### **Logging Integration**
- **Weights & Biases**: Comprehensive experiment tracking
- **TensorBoard**: Real-time training visualization
- **Custom Metrics**: Goal vector analysis, hierarchical-specific metrics

## **Development**

### **Extending the Agent**
```python
# Custom planner configuration
planner_config = {
    'hidden_size': 512,
    'num_layers': 3,
    'slstm_ratio': 0.7,
    'dropout': 0.1
}

# Custom controller configuration
controller_config = {
    'hidden_sizes': [512, 256, 128],
    'use_attention': True,
    'use_goal_conditioning': 'film'
}

# Create agent
agent = Andromeda2Agent(
    observation_size=107,
    planner_config=planner_config,
    controller_config=controller_config
)
```

### **Custom Reward Functions**
Modify intrinsic reward weights in configuration:
```yaml
intrinsic_rewards:
  car_velocity: 1.0
  ball_velocity: 1.0  
  car_to_ball_pos: 1.5
  ball_to_goal_pos: 2.0
```

### **xLSTM Configuration**
Customize the official xLSTM architecture:
```yaml
planner:
  hidden_size: 512
  num_layers: 4
  slstm_at_layer: [0, 2]      # sLSTM for temporal tracking
  mlstm_at_layer: [1, 3]      # mLSTM for strategic memory
  mlstm_num_heads: 8          # Attention heads
  context_length: 4096        # Longer memory
```


### **Debug Mode**
Enable debug features in configuration:
```yaml
debug:
  profile_performance: true
  verbose_logging: true
  check_numerics: true
  plot_gradients: true
```

### **Performance Optimization**
- Use GPU with sufficient VRAM (8GB+ recommended)
- Optimize `num_envs` based on available CPU cores
- Enable mixed precision training for faster convergence
- Use vectorized environments for maximum throughput

## **Future Development (Phase 2)**

### **Latent Goal Space**
Move from physical goal vectors to learned latent representations:
- Emergent strategic language between planner and controller
- Goal discriminator for unsupervised strategy discovery
- Higher-level strategic abstractions

### **Advanced Features**
- Multi-agent coordination strategies
- Opponent modeling and adaptation
- Transfer learning across different game modes
- Real-time strategy adaptation
