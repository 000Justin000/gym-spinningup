import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from PIL import Image

# --- Hyperparameters ---
EPISODES = 5000
LR = 1e-4
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1e5  # decay over 100k steps
BATCH_SIZE = 32
MEMORY_SIZE = 100_000
TARGET_UPDATE = 1000  # steps
STACK_SIZE = 4  # frames stacked
DEVICE = "mps"


# --- Preprocess function ---
def preprocess(obs):
    # Convert to grayscale and resize to 84x84 using PIL
    img = Image.fromarray(obs)
    img = img.convert("L")  # Convert to grayscale
    img = img.resize((84, 84), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        return (
            torch.tensor(np.stack(state), dtype=torch.float32).to(DEVICE) / 255.0,
            torch.tensor(action, dtype=torch.long).to(DEVICE),
            torch.tensor(reward, dtype=torch.float32).to(DEVICE),
            torch.tensor(np.stack(next_state), dtype=torch.float32).to(DEVICE) / 255.0,
            torch.tensor(done, dtype=torch.float32).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# --- CNN DQN Model ---
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(STACK_SIZE, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        return self.net(x)


# --- Action selection ---
def select_action(state, policy_net, steps_done):
    eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1.0 * steps_done / EPS_DECAY)
    if random.random() < eps:
        return random.randrange(n_actions)
    else:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE) / 255.0
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).item()


# --- Stack frames ---
def stack_frames(frames, state):
    frames.append(preprocess(state))
    while len(frames) < STACK_SIZE:
        frames.append(frames[-1])
    return np.stack(frames, axis=0)


# --- Setup environment ---
env = gym.make("MsPacman-v4", render_mode=None)
n_actions = env.action_space.n
buffer = ReplayBuffer(MEMORY_SIZE)

policy_net = DQN((STACK_SIZE, 84, 84), n_actions).to(DEVICE)
target_net = DQN((STACK_SIZE, 84, 84), n_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
steps_done = 0

# --- Training Loop ---
for episode in range(EPISODES):
    state, _ = env.reset()
    frames = deque(maxlen=STACK_SIZE)
    stacked_state = stack_frames(frames, state)

    total_reward = 0
    done = False

    while not done:
        action = select_action(stacked_state, policy_net, steps_done)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        next_stacked_state = stack_frames(frames, next_state)
        buffer.push(stacked_state, action, reward, next_stacked_state, done)
        stacked_state = next_stacked_state
        steps_done += 1

        # Train step
        if len(buffer) > BATCH_SIZE:
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            next_q = target_net(next_states).max(1)[0].detach()
            expected_q = rewards + GAMMA * next_q * (1 - dones)

            loss = nn.functional.mse_loss(q_values, expected_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update target network
        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}: Total reward = {total_reward}")

env.close()
