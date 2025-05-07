import gym
from time import sleep

# env = gym.make("CartPole-v0", render_mode="human")
# env = gym.make("MsPacman-v4", render_mode="human")
env = gym.make("MsPacmanNoFrameskip-v4", render_mode="human")
# env = gym.make("ALE/Pacman-v5", render_mode="human")
env.reset()
env.render()

for _ in range(1000):
    action = env.action_space.sample()
    ob_next, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        env.reset()
    sleep(0.01)

print("done")