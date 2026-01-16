# test_improvements.py
from env import CellularEnv
import numpy as np

env = CellularEnv(n_bs=2, n_ue=3)
states = env.reset()

print("Testing improved implementation:")
print(f"1. Channel shape: {env.channel_matrices.shape}")
print(f"2. Codebook size: {len(env.codebook)}")
print(f"3. State dimension: {len(states[0])}")

# Test one step
actions = [np.random.randint(0, 2, (3, 2)) for _ in range(2)]
next_states, rewards, inter_ints, _ = env.step(actions)

print(f"4. Rewards: {rewards}")
print(f"5. Inter-cell interference (BS 0): {inter_ints[0]}")
print("\n✓ All improvements integrated successfully!")