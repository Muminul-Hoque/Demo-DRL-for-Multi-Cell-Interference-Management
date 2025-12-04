"""
Simple Multi-Cell Interference Environment for DRL
Simulates 3 base stations with interference management

Author: Muhammed Muminul Hoque
Purpose: Preliminary work for federated MARL research with Prof. Vaezi
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MultiCellInterferenceEnv(gym.Env):
    """
    A simplified multi-cell interference environment.
    
    - 3 Base Stations (BS)
    - Each BS can choose power level: {0.1, 0.3, 0.5, 1.0}
    - Reward = sum of SINR across all BSs
    - Interference from neighboring BSs affects SINR
    """
    
    def __init__(self, n_cells=3):
        super(MultiCellInterferenceEnv, self).__init__()
        
        self.n_cells = n_cells
        self.power_levels = [0.1, 0.3, 0.5, 1.0]  # Discrete power choices
        
        # Action space: each cell chooses a power level
        self.action_space = spaces.Discrete(len(self.power_levels))
        
        # Observation space: [channel_gains, interference_levels, current_powers]
        # Shape: (n_cells * 3,) - simplified state representation
        self.observation_space = spaces.Box(
            low=0.0, high=2.0, 
            shape=(n_cells * 3,), 
            dtype=np.float32
        )
        
        # Initialize channel gains (random but stable per episode)
        self.channel_gains = None
        self.current_powers = None
        self.step_count = 0
        self.max_steps = 100
        
    def reset(self):
        """Reset environment to initial state"""
        # Random channel gains for each cell (distance-based path loss simulation)
        self.channel_gains = np.random.uniform(0.3, 1.0, self.n_cells)
        
        # Initialize with low power
        self.current_powers = np.ones(self.n_cells) * self.power_levels[0]
        
        self.step_count = 0
        
        return self._get_observation()
    
    def _get_observation(self):
        """Construct observation vector"""
        # Calculate current interference each cell receives
        interference = self._calculate_interference()
        
        # Observation: [channel_gains | interference | current_powers]
        obs = np.concatenate([
            self.channel_gains,
            interference,
            self.current_powers
        ])
        
        return obs.astype(np.float32)
    
    def _calculate_interference(self):
        """Calculate interference each cell receives from others"""
        interference = np.zeros(self.n_cells)
        
        for i in range(self.n_cells):
            # Sum interference from all other cells
            for j in range(self.n_cells):
                if i != j:
                    # Simplified interference model
                    # In reality: interference depends on distance, angles, etc.
                    interference_coefficient = 0.3  # Simplified
                    interference[i] += interference_coefficient * self.current_powers[j]
        
        return interference
    
    def _calculate_sinr(self):
        """Calculate Signal-to-Interference-plus-Noise Ratio for each cell"""
        interference = self._calculate_interference()
        noise_power = 0.01  # Background noise
        
        sinr = np.zeros(self.n_cells)
        for i in range(self.n_cells):
            signal = self.channel_gains[i] * self.current_powers[i]
            sinr[i] = signal / (interference[i] + noise_power)
        
        return sinr
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        For multi-agent: action would be a list of actions (one per BS)
        For single-agent: action controls one BS, others use fixed policy
        
        Here we use single-agent controlling BS 0, others maintain current power
        """
        # Update power for the controlled BS (BS 0)
        self.current_powers[0] = self.power_levels[action]
        
        # Other BSs maintain their power (in real multi-agent, they'd also choose)
        # For simplicity, let's have them randomly adjust
        for i in range(1, self.n_cells):
            # Simple heuristic: slightly adjust based on interference
            interference = self._calculate_interference()
            if interference[i] > 0.5:
                self.current_powers[i] = max(0.1, self.current_powers[i] * 0.9)
            else:
                self.current_powers[i] = min(1.0, self.current_powers[i] * 1.1)
        
        # Calculate SINR for all cells
        sinr = self._calculate_sinr()
        
        # Reward: sum of log(1 + SINR) - common metric in wireless
        reward = np.sum(np.log(1 + sinr))
        
        # Check if episode is done
        self.step_count += 1
        done = (self.step_count >= self.max_steps)
        
        # Get next observation
        obs = self._get_observation()
        
        # Info (for debugging)
        info = {
            'sinr': sinr,
            'powers': self.current_powers.copy(),
            'avg_sinr': np.mean(sinr)
        }
        
        return obs, reward, done, info
    
    def render(self, mode='human'):
        """Print current state (for debugging)"""
        sinr = self._calculate_sinr()
        print(f"Step {self.step_count}:")
        print(f"  Powers: {self.current_powers}")
        print(f"  SINR: {sinr}")
        print(f"  Avg SINR: {np.mean(sinr):.3f}")