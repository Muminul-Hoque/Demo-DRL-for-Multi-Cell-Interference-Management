"""
Simple DQN Agent for Multi-Cell Interference Management
Baseline implementation to demonstrate DRL feasibility

Author: Muhammed Muminul Hoque
Purpose: Preliminary work for federated MARL research with Prof. Vaezi
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from environment import MultiCellInterferenceEnv

class DQN(nn.Module):
    """Simple Deep Q-Network"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    """DQN Agent with Experience Replay"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-networks
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64
        
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """Train on a batch from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss and backprop
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with current Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

def train_dqn(n_episodes=500, target_update_freq=10):
    """Train DQN agent on interference environment"""
    # Create environment
    env = MultiCellInterferenceEnv(n_cells=3)
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    
    # Training metrics
    episode_rewards = []
    episode_avg_sinr = []
    losses = []
    
    print("Starting DQN Training...")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    print("-" * 50)
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = []
        sinr_values = []
        
        done = False
        while not done:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train
            loss = agent.train_step()
            if loss > 0:
                episode_loss.append(loss)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            sinr_values.append(info['avg_sinr'])
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Update target network periodically
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_avg_sinr.append(np.mean(sinr_values))
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_sinr = np.mean(episode_avg_sinr[-50:])
            print(f"Episode {episode+1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg SINR: {avg_sinr:.3f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    print("-" * 50)
    print("Training complete!")
    
    return agent, episode_rewards, episode_avg_sinr, losses

def plot_results(episode_rewards, episode_avg_sinr, save_path='results/training_results.png'):
    """Plot training results"""
    import os
    os.makedirs('results', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot rewards
    ax1.plot(episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    # Moving average
    window = 20
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 
                color='red', linewidth=2, label=f'{window}-Episode Moving Avg')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('DQN Training: Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot average SINR
    ax2.plot(episode_avg_sinr, alpha=0.3, color='green', label='Avg SINR')
    if len(episode_avg_sinr) >= window:
        moving_avg_sinr = np.convolve(episode_avg_sinr, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_avg_sinr)), moving_avg_sinr,
                color='darkgreen', linewidth=2, label=f'{window}-Episode Moving Avg')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average SINR')
    ax2.set_title('DQN Training: Network Performance (SINR)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Results saved to {save_path}")
    plt.close()

def evaluate_agent(agent, env, n_episodes=10):
    """Evaluate trained agent"""
    print("\nEvaluating trained agent...")
    eval_rewards = []
    eval_sinr = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        sinr_values = []
        
        done = False
        while not done:
            action = agent.select_action(state, training=False)  # No exploration
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            sinr_values.append(info['avg_sinr'])
            state = next_state
        
        eval_rewards.append(episode_reward)
        eval_sinr.append(np.mean(sinr_values))
    
    print(f"Evaluation Results (n={n_episodes} episodes):")
    print(f"  Average Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"  Average SINR: {np.mean(eval_sinr):.3f} ± {np.std(eval_sinr):.3f}")
    print("-" * 50)

if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Train agent
    agent, rewards, sinr_values, losses = train_dqn(n_episodes=500)
    
    # Plot results
    plot_results(rewards, sinr_values)
    
    # Evaluate
    env = MultiCellInterferenceEnv(n_cells=3)
    evaluate_agent(agent, env, n_episodes=10)


