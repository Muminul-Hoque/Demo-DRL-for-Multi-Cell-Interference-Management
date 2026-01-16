# src/dqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),  # Larger first layer
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.01, gamma=0.995):
        """
        Paper-accurate DQN agent
        
        Args:
            state_dim: State dimension (4 * n_ue for paper)
            action_dim: 2^(2*n_ue) - all possible joint actions
            lr: Learning rate η = 0.01 (paper Section IV.A)
            gamma: Discount factor α = 0.995 (paper Section IV.A)
        """
        self.q = DQN(state_dim, action_dim)
        self.target_q = DQN(state_dim, action_dim)
        self.target_q.load_state_dict(self.q.state_dict())
        
        self.optim = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.action_dim = action_dim
    
    def act(self, state, eps, n_ue):
        """
        Select joint action for all UEs
        
        Args:
            state: Current state vector
            eps: Exploration probability
            n_ue: Number of UEs (needed to decode action)
        
        Returns:
            actions: (n_ue, 2) binary matrix
                     actions[u] = [power_bit, beam_bit]
        """
        if random.random() < eps:
            # Random exploration: sample random binary action matrix
            return np.random.randint(0, 2, (n_ue, 2))
        
        # Exploitation: select best action from Q-network
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q(state)
        
        # Get action index with highest Q-value
        action_idx = q_values.argmax().item()
        
        # Decode scalar index to binary action matrix
        return self._decode_action(action_idx, n_ue)
    
    def _decode_action(self, action_idx, n_ue):
        """
        Correctly decode scalar action to binary matrix.
        For n_ue=3: action_idx ∈ [0, 63]
        Returns: (3, 2) binary matrix
        """
        actions = np.zeros((n_ue, 2), dtype=int)
        for u in range(n_ue):
            # Extract 2 bits for UE u
            actions[u, 0] = (action_idx >> (2*u)) & 1       # Power bit
            actions[u, 1] = (action_idx >> (2*u + 1)) & 1   # Beam bit
        return actions
    
    def _encode_action(self, actions):
        """
        Convert binary action matrix to scalar index
        
        Args:
            actions: (n_ue, 2) binary matrix
        
        Returns:
            action_idx: Scalar in [0, 2^(2*n_ue) - 1]
        """
        action_idx = 0
        n_ue = actions.shape[0]
        for u in range(n_ue):
            action_idx |= (int(actions[u, 0]) << (2*u))
            action_idx |= (int(actions[u, 1]) << (2*u + 1))
        return action_idx
    
    def update(self, batch):
        """
        Update Q-network using DQN loss (Equation 15 in paper)
        
        L(θt) = (1/B) * Σ_b (yb - Q(sb, ab; θt))²
        where yb = rb + α * max_a' Q(s'b, a'b; θt-1)
        
        Args:
            batch: List of (state, action, reward, next_state) tuples
        
        Returns:
            loss: Scalar loss value
        """
        states, actions, rewards, next_states = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        # Encode action matrices to scalar indices for gathering Q-values
        actions = torch.LongTensor([self._encode_action(a) for a in actions]).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        
        # Q(s, a; θt) - current Q-values for taken actions
        q_vals = self.q(states).gather(1, actions).squeeze()
        
        # Target: y = r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_q(next_states).max(1)[0]
            target = rewards + self.gamma * next_q
        
        # MSE loss (Equation 15)
        loss = nn.MSELoss()(q_vals, target)
        
        # Gradient descent
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        return loss.item()
    
    def sync_target(self):
        """Copy Q-network weights to target network"""
        self.target_q.load_state_dict(self.q.state_dict())

    def get_enhanced_state(self, env, l):
        """
        Enhanced state representation:
        For each UE: [norm_power, norm_beam, norm_x, norm_y]
        """
        state = []
        for u in range(env.n_ue):
            # Normalize power to [0, 1]
            power_norm = (env.prev_powers_dbm[l, u] - env.min_power_dbm) / \
                        (env.max_power_dbm - env.min_power_dbm)
            
            # Normalize beam index
            beam_norm = env.prev_beams[l, u] / env.n_beams
            
            # Normalize position (relative to cell center)
            x_norm = env.ue_positions[l, u, 0] / env.cell_radius
            y_norm = env.ue_positions[l, u, 1] / env.cell_radius
            
            state.extend([power_norm, beam_norm, x_norm, y_norm])
        
        return np.array(state, dtype=np.float32)