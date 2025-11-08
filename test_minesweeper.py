from minesweeper_env import MinesweeperEnv
import random

env = MinesweeperEnv(size=5, n_mines=3)

obs = env.reset()
env.render()

done = False
total_reward = 0

while not done:
    hidden_cells = [(x, y) for x in range(env.size) for y in range(env.size) if not env.visible[x, y]]
    action = random.choice(hidden_cells)
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    env.render()
    print(f"Action: {action}, Reward: {reward}, Done: {done}\n")

print(f"Total Reward: {total_reward}")
