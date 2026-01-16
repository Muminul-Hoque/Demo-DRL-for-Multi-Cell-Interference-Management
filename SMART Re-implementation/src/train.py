# src/train.py
from env import CellularEnv
from buffer import ReplayBuffer
from dqn import DQNAgent
import numpy as np
import random

# Paper parameters (EXACT from paper)
N_BS = 2
N_UE = 3
EPISODES = 8000
STEPS = 50
BATCH = 32
GAMMA_MIN_DB = -3  # Paper value
I_MIN_DBM = -110    # Paper value (noise floor)

# Initialize environment
env = CellularEnv(
    n_bs=N_BS, 
    n_ue=N_UE,
    gamma_min_db=GAMMA_MIN_DB,
    i_min_dbm=I_MIN_DBM
)

state_dim = 4 * N_UE
action_dim = 2 ** (2 * N_UE)

agents = [DQNAgent(state_dim, action_dim, lr=0.01, gamma=0.995) 
          for _ in range(N_BS)]

local_buffers = [ReplayBuffer(capacity=10000) for _ in range(N_BS)]
shared_buffers = [ReplayBuffer(capacity=10000) for _ in range(N_BS)]

eps = 1.0
eps_decay = 0.995
eps_min = 0.1

sum_rates = []
shared_count = 0
total_experiences = 0
positive_rewards = 0

print("Starting SMART training...")
print(f"Cells: {N_BS}, UEs per cell: {N_UE}, Episodes: {EPISODES}")
print(f"State dim: {state_dim}, Action dim: {action_dim}")
print(f"Gamma_min: {10**(GAMMA_MIN_DB/10):.4f} ({GAMMA_MIN_DB} dB)")
print(f"I_min: {10**(I_MIN_DBM/10)/1000:.2e} W ({I_MIN_DBM} dBm)")
print(f"Noise floor: {10**(I_MIN_DBM/10)/1000:.2e} W ({I_MIN_DBM} dBm)")
print(f"Antennas per BS: {env.M}, Codebook size: {env.n_beams}")
print("="*60)

for ep in range(EPISODES):
    states = env.reset()
    ep_reward = 0
    ep_shared = 0
    ep_positive = 0
    
    for t in range(STEPS):
        # Select actions
        actions = []
        for l in range(N_BS):
            action_l = agents[l].act(states[l], eps, N_UE)
            actions.append(action_l)
        
        # Execute
        next_states, rewards, inter_cell_ints, done = env.step(actions)
        
        # Track positive rewards
        for r in rewards:
            if r > 0:
                ep_positive += 1
        
        # Experience sharing and storage
        # CORRECTED sharing logic (Eq. 14)
        # Experience sharing (SMART Eq. 14 – corrected)
        for l in range(N_BS):
            transition = (states[l], actions[l], rewards[l], next_states[l])
            local_buffers[l].add(transition)
            total_experiences += 1

            # Check if this transition should be shared
            for j in range(N_BS):
                if j == l:
                    continue

                share_flag = False
                for u in range(N_UE):
                    I_from_l_to_j = env.get_interference_from_to(l, j, u)

                    # SMART condition: high interference + negative reward
                    if (I_from_l_to_j > env.i_min) and (rewards[l] < 0):
                        share_flag = True
                        break

                if share_flag:
                    shared_buffers[j].add(transition)
                    ep_shared += 1
                    shared_count += 1
                    break  # share ONLY to the affected BS

                    
            
            
            # Training
            combined_buffer = list(local_buffers[l].buffer) + list(shared_buffers[l].buffer)
            
            if len(combined_buffer) >= BATCH:
                batch = random.sample(combined_buffer, BATCH)
                loss = agents[l].update(batch)
        
        states = next_states
        ep_reward += np.mean(rewards)
    
    # Count positive rewards in episode
    if ep_positive > 0:
        positive_rewards += 1
    
    # Decay exploration
    eps = max(eps_min, eps * eps_decay)
    
    # Update target networks
    if ep % 20 == 0:
        for agent in agents:
            agent.sync_target()
    
    # Logging
    sum_rates.append(ep_reward)
    
    if ep % 100 == 0:
        avg_rate = np.mean(sum_rates[-100:]) / STEPS if len(sum_rates) >= 100 else np.mean(sum_rates)
        share_ratio = shared_count / max(total_experiences, 1) * 100
        positive_ratio = positive_rewards / max(ep + 1, 1) * 100
        
        print(f"Episode {ep:4d} | Avg Sum-Rate: {avg_rate:8.2f} | "
              f"Eps: {eps:.3f} | Sharing: {share_ratio:5.1f}% | "
              f"Positive: {positive_ratio:5.1f}%")
        
        # Debug info for first few episodes
        if ep <= 200:
            print(f"  └─ Last episode reward: {sum_rates[-1]:.2f}, "
                  f"Shared: {ep_shared}/{STEPS*N_BS} experiences")

print("\n" + "="*60)
print("Training completed!")
print(f"Final per-step sum-rate: {np.mean(sum_rates[-100:])/STEPS:.2f} bps/Hz")
print(f"Total sharing ratio: {shared_count/total_experiences*100:.1f}%")
print(f"Episodes with positive rewards: {positive_rewards}/{EPISODES} ({positive_rewards/EPISODES*100:.1f}%)")
print(f"\nExpected results from paper:")
print(f"  - Sum-rate: ~6-8 bps/Hz")
print(f"  - Sharing ratio: ~25% (75% not shared)")
print(f"  - Most episodes should have positive rewards after convergence")
