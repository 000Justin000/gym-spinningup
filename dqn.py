import gym
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack

# Check CUDA availability (GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Environment setup
env_id = "MsPacman-v4"
# Create vectorized Atari environment with frame stacking (e.g., 4 frames for temporal information)
env = make_atari_env(env_id, n_envs=1, seed=0)
env = AtariWrapper(env)
env = VecFrameStack(env, n_stack=4)

# Model configuration
model = DQN(
    policy="CnnPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=50000,
    batch_size=32,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    verbose=1,
    tensorboard_log="./dqn_mspacman_tensorboard/",
    device=device  # Use GPU if available
)

# Train the model (1 million timesteps is standard for Atari games, adjust as needed)
timesteps = 1_000_000
model.learn(total_timesteps=timesteps, log_interval=100)

# Save the trained model
model.save("dqn_mspacman")

# Load the model (optional)
model = DQN.load("dqn_mspacman", env=env, device=device)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10, deterministic=True)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Watch the trained agent play
obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()