import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from utils import traci_helpers as th


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.model(x)


class JunctionDQNAgent:
    def __init__(self, tls_id, state_dim, action_dim, epsilon=0.1, gamma=0.99, lr=0.001):
        self.tls_id = tls_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Memory for basic experience replay
        self.memory = []

        self.max_memory = 10000
        self.batch_size = 32

    def get_state(self):
        lanes = th.get_incoming_lanes(self.tls_id)
        state = []
        for lane in lanes:
            state.append(th.get_lane_queue_length(lane))
            state.append(th.get_lane_waiting_time(lane))
        return np.array(state, dtype=np.float32)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values).item()

    def apply_action(self, action):
        # Each action maps to a phase
        th.set_traffic_light_phase(self.tls_id, action)

    def store_experience(self, state, action, reward, next_state):
        if len(self.memory) >= self.max_memory:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q_values = self.policy_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + self.gamma * max_next_q_values

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
