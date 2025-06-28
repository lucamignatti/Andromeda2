# **Andromeda2**

## **1\. Project Overview**

This project aims to develop a high-performance Rocket League agent by moving beyond purely reactive control. The core objective is to create an agent with a genuine capacity for strategic planning and long-term decision-making. We will achieve this by implementing a **Hierarchical Reinforcement Learning (HRL)** architecture that explicitly separates the "brain" (strategic planning) from the "muscles" (mechanical control).

## **2\. Core Philosophy: Separating Strategy from Mechanics**

Traditional RL agents often learn a single, monolithic policy that maps game states directly to actions. This can lead to mechanically impressive but strategically brittle behavior. Our approach is fundamentally different:

* **The Planner (The "Brain"):** A high-level policy that operates on a slower, more deliberate timescale. It processes the history of the game to understand the strategic landscape and formulates a high-level plan or *intention*.  
* **The Controller (The "Muscles"):** A low-level policy that operates in real-time. Its sole job is to execute the current plan provided by the Planner, translating strategic intent into precise, mechanical actions.

This separation allows each component to specialize, leading to a more robust, interpretable, and powerful agent.

## **3\. Architectural Design**

Our agent is a hybrid system composed of two distinct neural network modules.

### **3.1. The Planner Core**

* **Architecture:** **Extended Long Short-Term Memory (xLSTM)**.  
* **Function:** This recurrent network ingests sequences of game states from RLGym. Its powerful memory structures (sLSTM for state-tracking, mLSTM for high-capacity memory) allow it to build a rich, temporally-aware internal representation—the "latent brain space."  
* **Output:** A low-dimensional **Goal Vector** that represents the current strategic plan.

### **3.2. The Motor Control Head**

* **Architecture:** A simple **Multi-Layer Perceptron (MLP)**.  
* **Function:** This fast, non-recurrent network ensures real-time responsiveness.  
* **Inputs:** It takes two inputs at every game tick:  
  1. The current, immediate game state vector.  
  2. The current Goal Vector provided by the Planner.  
* **Output:** The low-level, continuous control actions (throttle, steer, pitch, yaw, etc.) required to execute the plan.

## **4\. Training Methodology**

We will use an on-policy HRL approach with **Proximal Policy Optimization (PPO)**. A version of PPO that supports recurrent policies is required to correctly manage the xLSTM's hidden state during training rollouts.

The key to this methodology is a two-tiered reward system:

| Component | Reward Signal | Purpose |
| :---- | :---- | :---- |
| **Planner (xLSTM)** | **Extrinsic Reward** | Learns to win the game. It is rewarded *only* by events from the game environment (goals, saves, demos, conceding goals). It connects its Goal Vector commands to final game outcomes. |
| **Controller (MLP)** | **Intrinsic Reward** | Learns to be a perfect subordinate. It is rewarded *only* for how well it achieves the Goal Vector set by the Planner. It has no knowledge of the game's score or objectives. |

This dual-reward system allows the Planner to focus on high-level strategy without getting bogged down in mechanical details, while the Controller can efficiently master mechanics without being confused by complex game scenarios.

## **5\. Goal Vector Specification (Phase 1\)**

For the initial implementation, the Goal Vector will be a 12-dimensional vector representing a desired physical state. This provides a rich language for the Planner to express intent.

**Goal Vector g\_t:**

* **Target Car Velocity (3D):** \[car\_vel\_x, car\_vel\_y, car\_vel\_z\]  
* **Target Ball Velocity (3D):** \[ball\_vel\_x, ball\_vel\_y, ball\_vel\_z\]  
* **Target Car-to-Ball Relative Position (3D):** \[car\_to\_ball\_x, y, z\]  
* **Target Ball-to-Opponent-Goal Relative Position (3D):** \[ball\_to\_opp\_goal\_x, y, z\]

### **Controller's Intrinsic Reward Function**

The Controller's reward, R\_intrinsic, is the negative squared error between the actual state and the goal state, encouraging it to minimize the difference.

R\_intrinsic \=  
\- w\_cv \* || actual\_car\_vel \- target\_car\_vel ||²  
\- w\_bv \* || actual\_ball\_vel \- target\_ball\_vel ||²  
\- w\_cp \* || actual\_car\_to\_ball\_pos \- target\_car\_to\_ball\_pos ||²  
\- w\_bp \* || actual\_ball\_to\_opp\_goal\_pos \- target\_ball\_to\_opp\_goal\_pos ||²  
The w\_... terms are tunable hyperparameters that balance the importance of each goal component.

## **6\. Hyperparameter Tuning & Evaluation**

Success in RL requires rigorous tuning and evaluation.

* **Key Hyperparameters:**  
  * RL parameters: learning\_rate, batch\_size, context\_length.  
  * xLSTM architecture: Ratio of sLSTM to mLSTM blocks, embedding\_dimension.  
  * Intrinsic Reward: The w weights in the reward function.  
* **Evaluation Protocol:**  
  1. Use modern Hyperparameter Optimization (HPO) methods like Bayesian Optimization.  
  2. Define disjoint sets of random seeds for tuning and final testing to prevent overfitting.  
  3. The primary metric is **win rate** against a suite of benchmark opponents (Psyonix bots, RLBot community bots of varying skill, and a frozen version of our own agent from previous checkpoints).

## **7\. Future Work and Exploration (Phase 2\)**

Once the Phase 1 agent is stable and performing well, the next step is to increase the strategic expressiveness of the system by moving to a more abstract goal representation.

### **The Latent Goal Space**

* **Concept:** Instead of a physically-grounded goal vector, the Planner will output a low-dimensional **latent goal vector** (e.g., 8-16 dimensions) that has no predefined meaning.  
* **Emergent Language:** The Planner and Controller must co-adapt to create a shared, implicit "language." The Planner learns to encode abstract strategies (e.g., "apply pressure," "play defensively," "set up an air dribble") into this vector, and the Controller learns to decode these abstract commands into sequences of actions.

### **Training with a Goal Discriminator**

This advanced approach requires a third network, a **Goal Discriminator (D)**.

1. **Planner (P):** Outputs a latent goal g.  
2. **Controller (C):** Takes actions to produce a sequence of states s.  
3. **Discriminator (D):** Is trained to predict whether a state s satisfies a given latent goal g.  
4. **Intrinsic Reward:** The Controller's reward is now the output of the Discriminator: R\_intrinsic \= D(s, g). The Controller is rewarded for producing states that "trick" the Discriminator into thinking it has achieved the Planner's abstract goal.

This method removes the need for hand-crafting complex intrinsic reward functions and allows for the emergence of far more nuanced and creative strategies.