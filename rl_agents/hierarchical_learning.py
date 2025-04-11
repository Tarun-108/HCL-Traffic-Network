import torch
from rl_agents.junction_agent import JunctionDQNAgent

class HierarchicalManager:
    def __init__(self, junction_ids, state_dim, action_dim):
        self.junction_agents = {
            j_id: JunctionDQNAgent(state_dim, action_dim, tls_id=j_id)
            for j_id in junction_ids
        }

    def act_all(self, state_dict):
        """
        state_dict: {junction_id: state_vector}
        Returns: {junction_id: selected_action}
        """
        actions = {}
        for j_id, agent in self.junction_agents.items():
            state = state_dict[j_id]
            actions[j_id] = agent.act(state)
        return actions

    def remember_all(self, transition_dict):
        """
        transition_dict: {
            junction_id: (state, action, reward, next_state, done)
        }
        """
        for j_id, transition in transition_dict.items():
            self.junction_agents[j_id].remember(*transition)

    def replay_all(self):
        for agent in self.junction_agents.values():
            agent.replay()

    def update_all_target_networks(self):
        for agent in self.junction_agents.values():
            agent.update_target_network()

    def save_all_models(self, base_path="results/models"):
        for j_id, agent in self.junction_agents.items():
            agent.save_model(f"{base_path}/agent_{j_id}.pth")

    def load_all_models(self, base_path="results/models"):
        for j_id, agent in self.junction_agents.items():
            agent.load_model(f"{base_path}/agent_{j_id}.pth")

    def get_weights(self):
        return {j_id: agent.policy_net.state_dict()
                for j_id, agent in self.junction_agents.items()}

    def average_weights(self, target_ids=None):
        if target_ids is None:
            target_ids = list(self.junction_agents.keys())

        # Assume all models have same architecture
        avg_weights = {}
        count = len(target_ids)

        for key in self.junction_agents[target_ids[0]].policy_net.state_dict():
            avg_weights[key] = sum(
                self.junction_agents[j_id].policy_net.state_dict()[key]
                for j_id in target_ids
            ) / count

        for j_id in target_ids:
            self.junction_agents[j_id].policy_net.load_state_dict(avg_weights)
            self.junction_agents[j_id].update_target_network()

