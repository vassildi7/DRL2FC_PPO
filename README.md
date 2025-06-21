# DRL2FC_PPO
This project implements a Proximal Policy Optimization (PPO) agent to control frequency in a single-area Load Frequency Control (LFC) system. The environment is modeled in a custom Gymnasium-compatible simulator. The agent is trained using Stable-Baselines3 and demonstrates strong performance under randomly generated load disturbances.

---

## Project Overview

- **Environment**: Custom single-area LFC gym environment.
- **Agent**: Trained with PPO algorithm.
- **Goal**: Maintain frequency stability and minimize Area Control Error (ACE) and control effort under varying disturbances.

---

## Setup Instructions

### 1. Clone the Repository

### 2. Create Virtual Environment (Optional)

### 3. Install Dependencies

---

## Training the PPO Agent

python train_ppo.py

This will:
- Train using the custom LFC environment
- Save models to local directories
- Plot training episode rewards

---

## Testing & Evaluation

python test_ppo.py

This will:
- Load the saved model
- Simulate an LFC episode with disturbances
- Plot frequency, power, control signal, and reward over time

---

## Dependencies

pip install -r requirements.txt

This will install:
- gymnasium
- stable-baselines3
- numpy
- matplotlib

---

## Lisence
This project is licensed under the MIT License. See LICENSE for details.







