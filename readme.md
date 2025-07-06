# **Andromeda2: A Deliberative, Hierarchical Agent for Deep Reinforcement Learning**

Andromeda2 is a next-generation reinforcement learning agent designed to achieve a more human-like level of strategic planning and execution. It moves beyond purely reactive policies by integrating three core principles: **Hierarchical Control**, **Internal Simulation**, and **Deliberative Planning**.

This document serves as the technical blueprint for the project, detailing the agent's architecture, training methodology, and the key innovations that drive its performance.

---

## **1. Core Philosophy**

The design of Andromeda2 is guided by a vision of creating agents that don't just learn optimal actions, but also understand the *why* behind them.

1.  **Hierarchical Control (The "Mind and Body"):** We explicitly separate high-level strategic thinking from low-level motor control. A **Planner** decides *what* to do, and a **Controller** figures out *how* to do it. This mirrors cognitive models of human skill, separating conscious decision-making from muscle memory.

2.  **Internal Simulation (The "Imagination"):** The agent first learns a predictive **World Model** of its environment. This model acts as an internal "dream engine," allowing the agent to simulate future possibilities and understand the consequences of actions without having to experience them in the real world.

3.  **Deliberative Planning (The "Conscious Thought"):** The Planner is not just a reactive network. During training, it uses the World Model as a sandbox to propose multiple strategies, simulate their outcomes, and select the most promising one. This deliberative process allows for a deeper, more robust form of planning.

---

## **2. Architecture Deep Dive**

The agent is composed of four primary, interconnected components.

### **2.1. The World Model (The "Dream Engine")**
*   **Purpose:** To learn the temporal dynamics of the environment, enabling prediction and imagination.
*   **Style:** A Recurrent State-Space Model (RSSM), inspired by the Dreamer architecture.
*   **Components:**
    *   **Encoder (MLP):** Compresses the high-dimensional environment state `s_t` into a compact, probabilistic **latent state `z_t`**.
    *   **Dynamics Model (xLSTM):** The predictive core. It takes the previous latent state `z_{t-1}` and action `a_{t-1}` to predict the distribution of the current latent state `z_t`. This allows the model to "roll forward" a simulation.
    *   **Predictors (MLPs):** Two small networks that operate on latent states: a **Reward Predictor** to forecast future rewards and an **Observation Predictor (Decoder)** to reconstruct the environment state for training.

### **2.2. The Planner (The "Strategist")**
*   **Purpose:** To formulate high-level, long-term strategies.
*   **Architecture:** An autoregressive xLSTM core with a **Mixture Density Network (MDN)** head.
*   **Functionality:**
    *   The xLSTM body processes sequences of latent states from the World Model to build a strategic context.
    *   The MDN head outputs parameters for a **mixture of `k` Gaussian distributions**. This allows the Planner to represent multiple, distinct strategic options (e.g., "attack," "defend," "disrupt") simultaneously, along with its confidence in each.
    *   **Output:** A set of `k` potential **latent goal vectors (`g`)**, each representing a coherent high-level intention.

### **2.3. The Controller (The "Muscle Memory")**
*   **Purpose:** To execute the Planner's chosen strategy with low-level motor actions.
*   **Architecture:** A small, fast MLP.
*   **Functionality:** It receives the current environment state `s_t` and the single, final latent goal `g_final` selected by the Planner. Its sole job is to produce the continuous action `a_t` that best achieves this goal in the immediate future.

### **2.4. The Critics (The "Evaluators")**
*   **Purpose:** To provide the learning signals that train the Planner and Controller.
*   **Components:**
    *   **Extrinsic Critic (Temporal C51):** A sophisticated distributional critic that evaluates the long-term, game-winning potential of a state. It predicts not just a single value, but a full probability distribution over future rewards, and does so for multiple time horizons (e.g., 2, 5, and 10 seconds). This provides a rich, time-aware signal for training the Planner.
    *   **Intrinsic Critic (Value-Based):** A simple MLP critic that provides a dense, short-term reward signal to the Controller. It predicts the immediate success of the Controller in achieving the Planner's current goal `g`.

---

## **3. The Training Doctrine**

Training is a multi-stage process where the agent learns to dream, plan within that dream, and then act.

1.  **Phase A: World Model Training**
    *   The agent first collects real experience from the environment and uses it to train the **World Model**. The objective is to minimize the error in predicting future states and rewards, making the "dream engine" as accurate as possible.

2.  **Phase B: Imaginative Policy Learning**
    *   This is the core loop where the Planner and Controller are trained entirely within the World Model's imagination.
    *   **Step 1: Deliberation & Search:** At the start of a planning cycle, the **Planner** proposes its `k` candidate goals. For each candidate, a separate, short simulation is run using the World Model.
    *   **Step 2: Evaluation:** The **Temporal C51 Critic** evaluates each of the `k` simulated trajectories, returning a detailed temporal reward profile for each potential strategy.
    *   **Step 3: Selection:** A heuristic (e.g., selecting the strategy with the highest rate of reward increase, or "highest derivative") is applied to these profiles to select the single best goal, `g_best`.
    *   **Step 4: Planner Update:** The Planner is updated using a Mixture Density Network loss function. Its goal is to increase the probability of generating `g_best` in this situation, effectively learning from its own deliberation process.
    *   **Step 5: Controller Update:** The Controller is then trained for a short window, conditioned on `g_best`. It receives a dense intrinsic reward based on how well it makes the subsequent latent states match the goal `g_best`. This reward signal is provided by the **Intrinsic Critic**, which is updated simultaneously.

---

## **4. Inference-Time Policy**

For real-time performance, the slow, deliberative process is "distilled" into a fast, feed-forward policy.

**The World Model's simulation capability is DISABLED at inference time.**

1.  **Encode:** The current environment state `s_t` is encoded into a latent state `z_t`.
2.  **Propose:** The **Planner** takes `z_t` and outputs its `k` candidate goals (the means of its MDN components).
3.  **Evaluate:** The fast **Temporal C51 Critic** is used directly on the `k` goals to instantly predict their temporal reward profiles (no simulation required).
4.  **Select:** The same "highest derivative" heuristic chooses the final goal, `g_final`.
5.  **Act:** The **Controller** takes `s_t` and `g_final` to produce the final action `a_t`.

This creates an agent that benefits from the depth of a search-based process during training, while maintaining the speed of a reactive policy during deployment.

---

## **5. Key Innovations**

*   **MDN-based Planner:** Moves beyond simple goal-setting to represent a rich, multi-modal distribution of strategic intentions.
*   **Temporal Distributional Critic:** Provides a nuanced understanding of not just *how much* reward is expected, but *when* it is expected to occur.
*   **Training-Time Deliberation:** A novel training scheme where the agent uses its internal world model as a sandbox to test hypotheses before committing to a strategy.
*   **Decoupled Inference:** A clean separation between the slow training-time search and the fast real-time policy.

---

## **6. Project Structure & Workflow**

The project is structured as a series of independent stages, each with its own training and validation scripts. This ensures modularity and focused development.

```
/Andromeda2/
├── checkpoints/
│   ├── stage1_world_model/
│   ├── stage2_controller/
│   └── stage3_full_agent/
│
├── configs/
│   ├── training/
│   │   ├── stage1_world_model.yaml
│   │   ├── stage2_controller.yaml
│   │   └── stage3_full_agent.yaml
│   │
│   ├── controller.yaml
│   ├── extrinsic_critic.yaml
│   ├── global.yaml
│   ├── intrinsic_critic.yaml
│   ├── planner.yaml
│   └── world_model.yaml
│
├── src/
│   └── __init__.py
│
├── stage1_world_model/
│   ├── __init__.py
│   ├── train.py
│   └── validate.py
│
├── stage2_controller/
│   ├── __init__.py
│   ├── train.py
│   └── validate.py
│
├── stage3_full_agent/
│   ├── __init__.py
│   ├── train.py
│   └── validate.py
│
├── cleanup.py
├── environment.yml
├── README.md
└── .gitignore
```

### **Workflow:**

1.  **Setup:** Create the conda environment using `conda env create -f environment.yml`.
2.  **Stage 1: World Model:**
    *   Run `stage1_world_model/train.py`.
    *   Manually inspect the dream videos produced by `stage1_world_model/validate.py` to verify model quality.
3.  **Stage 2: Controller Pre-training:**
    *   Run `stage2_controller/train.py`, providing the path to the validated `latest.pt` world model checkpoint.
    *   Use `stage2_controller/validate.py` to get quantitative metrics on the controller's goal-achieving performance.
4.  **Stage 3: Full Agent Training:**
    *   Run `stage3_full_agent/train.py`, providing the validated checkpoints from the previous stages.
    *   Use `stage3_full_agent/validate.py` to measure the final agent's performance (e.g., win rate) in the actual environment.
5.  **Maintenance:** Periodically run `cleanup.py` to remove old, timestamped checkpoints and save disk space.
