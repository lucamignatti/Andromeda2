# **Andromeda2: A Deliberative, Hierarchical Agent for Deep Reinforcement Learning**

Andromeda2 is a next-generation reinforcement learning agent designed for complex, high-speed environments. It moves beyond purely reactive policies by integrating two core principles: **Hierarchical Control** and **Critic-Guided Deliberation**.

This document serves as the technical blueprint for the project, detailing the agent's architecture, training methodology, and the key innovations that drive its performance.

---

## **1. Core Philosophy**

The design of Andromeda2 is guided by a vision of creating agents that don't just learn optimal actions, but also understand the *why* behind them. This is achieved without the overhead of a predictive world model, making it ideal for fast-paced simulators like `rlgym`.

1.  **Hierarchical Control (The "Mind and Body"):** We explicitly separate high-level strategic thinking from low-level motor control. A **Planner** decides *what* to do, and a **Controller** figures out *how* to do it. This mirrors cognitive models of human skill, separating conscious decision-making from muscle memory.

2.  **Critic-Guided Deliberation (The "Educated Choice"):** The Planner is not just a reactive network. It learns to propose multiple, distinct strategies. These strategies are not simulated, but are instead rapidly evaluated by a learned **Temporal Critic**. The critic, trained on vast amounts of real game experience, provides a rich evaluation of each strategy's potential. This allows the agent to make deep, robust plans without the need for an internal simulation engine.

---

## **2. Architecture Deep Dive**

The agent is composed of four primary, interconnected components, all trained directly on environment data.

### **2.1. The State Encoder**
*   **Purpose:** To compress the high-dimensional environment state `s_t` into a compact, informative **encoded state `e_t`**.
*   **Architecture:** A standard Multi-Layer Perceptron (MLP).

### **2.2. The Planner (The "Strategist")**
*   **Purpose:** To formulate high-level, long-term strategies based on the current game context.
*   **Architecture:** An autoregressive xLSTM core with a **Mixture Density Network (MDN)** head.
*   **Functionality:**
    *   The xLSTM body processes sequences of encoded states `e_t` to build a strategic context.
    *   The MDN head outputs parameters for a **mixture of `k` Gaussian distributions**. This allows the Planner to represent multiple, distinct strategic options (e.g., "attack," "defend," "disrupt") simultaneously, along with its confidence in each.
    *   **Output:** A set of `k` potential **goal vectors (`g`)**, each representing a coherent high-level intention.

### **2.3. The Controller (The "Muscle Memory")**
*   **Purpose:** To execute the Planner's chosen strategy with low-level motor actions.
*   **Architecture:** A small, fast MLP.
*   **Functionality:** It receives the current encoded state `e_t` and the single, final goal `g_final` selected by the Planner. Its sole job is to produce the continuous action `a_t` that best achieves this goal in the immediate future.

### **2.4. The Critics (The "Evaluators")**
*   **Purpose:** To provide the learning signals that train the Planner and Controller.
*   **Components:**
    *   **Extrinsic Critic (Temporal C51):** A sophisticated distributional critic that evaluates the long-term, game-winning potential of a state-goal pair. It predicts not just a single value, but a full probability distribution over future rewards, and does so for multiple time horizons (e.g., 2, 5, and 10 seconds). This provides a rich, time-aware signal for training the Planner.
    *   **Intrinsic Critic (Value-Based):** A simple MLP critic that provides a dense, short-term reward signal to the Controller. It predicts the immediate success of the Controller in achieving the Planner's current goal `g`.

---

## **3. The Training Doctrine**

Training is a model-free, two-stage process that learns directly from experience collected from the environment.

1.  **Phase A: Controller Pre-training**
    *   The Controller is first pre-trained to achieve a wide variety of goals. This ensures it has a baseline level of competence before being integrated with the Planner.

2.  **Phase B: Full Agent Policy Learning**
    *   This is the core loop where the Planner and Controller are trained together on data from the real environment.
    *   **Step 1: Data Collection:** The agent interacts with the environment (`rlgym`) to collect a batch of trajectories.
    *   **Step 2: Deliberation & Evaluation:** For states in the collected batch, the **Planner** proposes its `k` candidate goals. The **Temporal C51 Critic** evaluates each of these `k` goals, returning a detailed temporal reward profile for each potential strategy.
    *   **Step 3: Selection:** A heuristic (e.g., selecting the strategy with the highest rate of reward increase, or "highest derivative") is applied to these profiles to select the single best goal, `g_best`.
    *   **Step 4: Planner Update:** The Planner is updated using a Mixture Density Network loss function. Its goal is to increase the probability of generating `g_best` in this situation, effectively learning from its own deliberation process.
    *   **Step 5: Controller & Critic Update:** The Controller is updated to better achieve `g_best`, and both critics are updated with the new data from the collected trajectories.

---

## **4. Inference-Time Policy**

The inference policy is identical to the training-time policy, making it fast, efficient, and robust.

1.  **Encode:** The current environment state `s_t` is encoded into `e_t`.
2.  **Propose:** The **Planner** takes `e_t` and outputs its `k` candidate goals.
3.  **Evaluate:** The fast **Temporal C51 Critic** is used directly on the `k` goals to instantly predict their temporal reward profiles.
4.  **Select:** The same "highest derivative" heuristic chooses the final goal, `g_final`.
5.  **Act:** The **Controller** takes `e_t` and `g_final` to produce the final action `a_t`.

This creates an agent that benefits from the depth of a search-like process while maintaining the speed of a purely reactive policy.

---

## **5. Key Innovations**

*   **MDN-based Planner:** Moves beyond simple goal-setting to represent a rich, multi-modal distribution of strategic intentions.
*   **Temporal Distributional Critic:** Provides a nuanced understanding of not just *how much* reward is expected, but *when* it is expected to occur.
*   **Critic-Guided Deliberation:** A novel training scheme where the agent uses its learned value function as a fast, effective proxy for simulation, allowing it to test hypotheses before committing to a strategy.

---

## **6. Training Framework**

To manage the complexity of the training process, we use a class-based training framework.

*   **`src/training/base_trainer.py`:** An abstract base class that defines the core interface for all trainers. It handles common tasks like loading global configurations and creating timestamped checkpoint directories.
*   **`src/training/controller_trainer.py`:** The trainer for Stage 1. It focuses on pre-training the Controller to achieve goals specified by the Planner.
*   **`src/training/full_agent_trainer.py`:** The trainer for Stage 2. It orchestrates the complex, deliberation-based training of the Planner and Controller together.

Each stage's main `train.py` and `validate.py` scripts will instantiate the corresponding trainer class and execute its `train()` or `validate()` method.

---

## **7. Project Structure & Workflow**

The project is structured as a series of independent stages, each with its own training and validation scripts. This ensures modularity and focused development.

```
/Andromeda2/
├── checkpoints/
│   └── ...
├── configs/
│   └── ...
├── src/
│   ├── components/
│   │   ├── ...
│   ├── training/
│   │   ├── __init__.py
│   │   ├── base_trainer.py
│   │   ├── controller_trainer.py
│   │   └── full_agent_trainer.py
│   ├── utils/
│   │   ├── ...
│   └── __init__.py
│
├── stage2_controller/
│   └── ...
├── stage3_full_agent/
│   └── ...
│
├── cleanup.py
├── environment.yml
├── README.md
└── .gitignore
```

### **Workflow:**

1.  **Setup:** Create the conda environment using `conda env create -f environment.yml`.
2.  **Stage 1: Controller Pre-training:**
    *   Run `stage2_controller/train.py` to execute the `ControllerTrainer`.
    *   Use `stage2_controller/validate.py` to get quantitative metrics on the controller's goal-achieving performance.
3.  **Stage 2: Full Agent Training:**
    *   Run `stage3_full_agent/train.py` to execute the `FullAgentTrainer`.
    *   Use `stage3_full_agent/validate.py` to measure the final agent's performance (e.g., win rate) in the actual environment.
4.  **Maintenance:** Periodically run `cleanup.py` to remove old, timestamped checkpoints and save disk space.
