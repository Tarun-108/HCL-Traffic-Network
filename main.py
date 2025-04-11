import os
import sys
import traci
import sumolib
import time
import csv
import matplotlib.pyplot as plt
import pandas as pd

from rl_agents.junction_agent import JunctionDQNAgent as Junction1Agent
from rl_agents.junction_agent import JunctionDQNAgent as Junction2Agent
from utils.traci_helpers import get_observation_for_tls as get_observation, get_reward_normalized as get_reward

# Configure SUMO
SUMO_BINARY = "sumo-gui"  # or "sumo"
NET_FILE = "sumo_network_4points/network.net.xml"
ROUTE_FILE = "sumo_network_4points_4points/routes.rou.xml"
ADDITIONAL_FILES = "sumo_network_4points/tls.add.xml"
SUMO_CMD = [SUMO_BINARY, "-n", NET_FILE, "-r", ROUTE_FILE, "-a", ADDITIONAL_FILES, "--step-length", "0.5", "--start"]

os.makedirs("results/logs", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

log_file_j1 = open("results/logs/j1_rewards.csv", mode="w", newline="")
log_file_j2 = open("results/logs/j2_rewards.csv", mode="w", newline="")

logger_j1 = csv.writer(log_file_j1)
logger_j2 = csv.writer(log_file_j2)

logger_j1.writerow(["Step", "Reward"])
logger_j2.writerow(["Step", "Reward"])

episode_metrics_j1 = open("results/logs/j1_episode_metrics.csv", mode="w", newline="")
episode_metrics_j2 = open("results/logs/j2_episode_metrics.csv", mode="w", newline="")

episode_logger_j1 = csv.writer(episode_metrics_j1)
episode_logger_j2 = csv.writer(episode_metrics_j2)

episode_logger_j1.writerow(["Episode", "TotalReward", "AvgReward", "Steps"])
episode_logger_j2.writerow(["Episode", "TotalReward", "AvgReward", "Steps"])

# Initialize simulation
def run_simulation(max_steps=1000):
    traci.start(SUMO_CMD)
    print("✅ SUMO simulation started.")
    state_dim = 6  # Number of lanes or state features
    action_dim = 4  # Number of traffic phases


    # Create agents
    agent_j1 = Junction1Agent(tls_id="A0", state_dim=state_dim, action_dim=action_dim)
    agent_j2 = Junction2Agent(tls_id="A1", state_dim=state_dim, action_dim=action_dim)

    for step in range(max_steps):
        traci.simulationStep()

        total_reward_j1 = 0
        total_reward_j2 = 0
        episode_steps = 0

        # J1 Agent
        obs_j1 = get_observation("A0")
        action_j1 = agent_j1.choose_action(obs_j1)
        agent_j1.apply_action(action_j1)
        reward_j1 = get_reward("A0")
        next_obs_j1 = get_observation("A0")
        agent_j1.store_experience(obs_j1, action_j1, reward_j1, next_obs_j1)

        # J2 Agent
        obs_j2 = get_observation("A1")
        action_j2 = agent_j2.choose_action(obs_j2)
        agent_j2.apply_action(action_j2)
        reward_j2 = get_reward("A1")
        next_obs_j2 = get_observation("A1")
        agent_j2.store_experience(obs_j2, action_j2, reward_j2, next_obs_j2)

        total_reward_j1 += reward_j1
        total_reward_j2 += reward_j2
        episode_steps += 1

        avg_reward_j1 = total_reward_j1 / episode_steps
        avg_reward_j2 = total_reward_j2 / episode_steps

        episode_logger_j1.writerow([episode_steps, total_reward_j1, avg_reward_j1, step])
        episode_logger_j2.writerow([episode_steps, total_reward_j2, avg_reward_j2, step])

        if step % 20 == 0:
            logger_j1.writerow([step, reward_j1])
            logger_j2.writerow([step, reward_j2])
            print(f"[Step {step}] J1 Reward: {reward_j1:.2f} | J2 Reward: {reward_j2:.2f}")

    traci.close()
    log_file_j1.close()
    log_file_j2.close()
    episode_metrics_j1.close()
    episode_metrics_j2.close()

    print("🛑 SUMO simulation ended.")

def plot_rewards(log_path, title, output_path):
    data = pd.read_csv(log_path)
    plt.plot(data["Step"], data["Reward"])
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    run_simulation()
    plot_rewards("results/logs/j1_rewards.csv", "Junction 1 Reward", "results/plots/j1_plot.png")
    plot_rewards("results/logs/j2_rewards.csv", "Junction 2 Reward", "results/plots/j2_plot.png")
