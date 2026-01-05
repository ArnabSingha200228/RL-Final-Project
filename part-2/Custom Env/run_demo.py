import os
import numpy as np
import gymnasium as gym
import register_envs
import pygame

from stable_baselines3 import SAC

eval_env = env = gym.make("BallPlate-v0", render_mode="human")

model = SAC.load("sac_ballplate", env=env)

episodes = 5
for ep in range(episodes):
    obs, _ = eval_env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = eval_env.step(action)

        total_reward += reward
        steps += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    print(f"[Render Eval] Episode {ep+1} | Total Reward: {total_reward:.2f} | Steps: {steps}")

eval_env.close()
