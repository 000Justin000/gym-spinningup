import gym
from time import sleep
import pygame
from pygame.locals import *

# env = gym.make("CartPole-v0", render_mode="human")
# env = gym.make("MsPacman-v4", render_mode="human")
env = gym.make("MsPacmanNoFrameskip-v4", render_mode="human")
# env = gym.make("ALE/Pacman-v5", render_mode="human")
env.reset()
env.render()

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((100, 100))  # Dummy window, required by pygame
pygame.display.set_caption("MsPacman Controller")

key_action_map = {
    K_SPACE: 0,
    K_UP: 1,
    K_RIGHT: 2,
    K_LEFT: 3,
    K_DOWN: 4
}

running = True
while running:
    # Handle events
    key_pressed = False
    while not key_pressed:
        for event in pygame.event.get():
            if event.type == QUIT:
                key_pressed = True
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_q:
                    key_pressed = True
                    running = False
                elif event.key in key_action_map:
                    action = key_action_map[event.key]
                    key_pressed = True
            if key_pressed:
                break
        sleep(0.01)
                # Apply action
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        observation = env.reset()