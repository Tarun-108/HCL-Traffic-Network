# Hierarchical Collaborative Learning for Traffic Management using SUMO

This project implements a Hierarchical Collaborative Learning (HCL) framework for optimizing traffic signal control across a simulated urban road network using the SUMO (Simulation of Urban Mobility) platform.

---

## 📁 Directory Structure

```text
project/
│
├── myenv/                          # [Excluded] Local virtual environment
│
├── RESULTS/                        # Stores logs and visualizations from experiments
│   ├── logs/
│   │   ├── j1_episode_metrics.csv
│   │   ├── j1_rewards.csv
│   │   ├── j2_episode_metrics.csv
│   │   └── j2_rewards.csv
│   └── plots/
│       ├── j1_plot.png
│       └── j2_plot.png
│
├── RL_AGENTS/                      # Reinforcement learning agents and logic
│   ├── hierarchical_learning.py   # Manages inter-agent learning hierarchy
│   ├── junction_agent.py          # DQN agent logic for each traffic junction
│   └── __pycache__/               # Compiled Python bytecode
│
├── SUMO_NETWORK_16POINTS/         # SUMO network with a 4x4 junction grid (16 nodes)
│   ├── detectors.add.xml
│   ├── network.edg.xml
│   ├── network.net.xml
│   ├── network.nod.xml
│   ├── routes.rou.xml
│   ├── simulation.sumocfg
│   └── tls.add.xml
│
├── SUMO_NETWORK_4POINTS/          # SUMO network with a 2x2 junction grid (4 nodes)
│   ├── network.net.xml
│   ├── routes.rou.xml
│   ├── simulation.sumocfg
│   └── tls.add.xml
│
└── UTILS/                         # Helper functions and wrappers
    ├── traci_helpers.py          # Utility functions for TraCI interaction
    └── __pycache__/              # Compiled Python bytecode

```

## 🧠 Project Overview
This project investigates multi-junction traffic signal optimization using a Hierarchical Collaborative Learning (HCL) approach, where each junction is controlled by a deep reinforcement learning agent (DQN), and knowledge is shared across levels (e.g., pairwise, regional, global) to enhance overall network throughput and reduce vehicle delay.

The system uses SUMO to simulate vehicle flows and TraCI (Traffic Control Interface) to enable real-time RL agent interaction with the simulation environment.

## 🧪 Components
RL_AGENTS: Implements Deep Q-Learning for junction-level optimization and hierarchical logic for collaboration.

UTILS: Provides wrappers for interacting with SUMO using TraCI.

SUMO_NETWORK_4POINTS / 16POINTS: Contain SUMO network definition files for two different topologies to evaluate scalability.

RESULTS: Stores training logs (reward curves, episode metrics) and performance visualizations for each agent.

## 🚦 Features
Reinforcement learning at individual junctions using DQN

Knowledge sharing across agents in hierarchical fashion

Multiple SUMO network configurations

Metrics and plots for evaluating performance

Modular and extensible codebase

## ▶️ Getting Started
Note: Ensure you have SUMO and Python 3.12+ installed with traci, numpy, matplotlib, and torch.

Activate the virtual environment (if using myenv):

```bash
.\myenv\Scripts\activate
```
Run the main simulation script (modify as needed):


```bash
python main.py
```
Check results in the RESULTS/logs/ and RESULTS/plots/ folders.


## 📈 Evaluation

Plots in the RESULTS/plots/ folder visualize how rewards and queue lengths evolve over time, validating the effectiveness of the HCL framework over isolated DQN agents.
