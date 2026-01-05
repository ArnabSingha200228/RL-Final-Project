
# without video


import gymnasium as gym
import gymnasium_robotics
import numpy as np
import time

from stable_baselines3 import PPO

# ===============================
# CONFIG
# ===============================
ENV_NAME = "FetchReach-v4"
MODEL_PATH = "ppo_fetchreach"   # without .zip

# ===============================
# LOAD MODEL
# ===============================
print("Loading trained PPO model...")
demo_env = gym.make(ENV_NAME, render_mode="human")

model = PPO.load(MODEL_PATH, env=demo_env)
print("Model loaded successfully.")

# ===============================
# DEMO RUN
# ===============================
obs, info = demo_env.reset()

for step in range(600):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = demo_env.step(action)

    achieved = obs["achieved_goal"]
    desired = obs["desired_goal"]
    dist = np.linalg.norm(achieved - desired)

    print(f"Step {step:03d} | Distance to goal: {dist:.4f}")

    time.sleep(0.01)

    if terminated or truncated:
        obs, info = demo_env.reset()

demo_env.close()
print("Demo finished.")




# ##with video 
# import gymnasium as gym
# import gymnasium_robotics
# import numpy as np
# import time
# import os

# from stable_baselines3 import PPO

# # ===============================
# # CONFIG
# # ===============================
# ENV_NAME = "FetchReach-v4"
# MODEL_PATH = "ppo_fetchreach"   # without .zip
# VIDEO_DIR = "videos"

# os.makedirs(VIDEO_DIR, exist_ok=True)

# # ===============================
# # ENV WITH VIDEO RECORDING
# # ===============================
# demo_env = gym.make(
#     ENV_NAME,
#     render_mode="rgb_array"
# )

# demo_env = gym.wrappers.RecordVideo(
#     demo_env,
#     video_folder=VIDEO_DIR,
#     episode_trigger=lambda episode_id: True,  # record every episode
#     name_prefix="PPO_FetchReach_Demo"
# )

# # ===============================
# # LOAD MODEL
# # ===============================
# print("Loading trained PPO model...")
# model = PPO.load(MODEL_PATH, env=demo_env)
# print("Model loaded successfully.")

# # ===============================
# # DEMO RUN
# # ===============================
# obs, info = demo_env.reset()

# for step in range(600):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = demo_env.step(action)

#     achieved = obs["achieved_goal"]
#     desired = obs["desired_goal"]
#     dist = np.linalg.norm(achieved - desired)

#     print(f"Step {step:03d} | Distance to goal: {dist:.4f}")

#     time.sleep(0.01)

#     if terminated or truncated:
#         obs, info = demo_env.reset()

# demo_env.close()
# print("Video saved in:", VIDEO_DIR)




# # drl_env\Scripts\activate


# # cd C:\DRL_FINAL_PROJECT\PPO
# # python run_demo_video.py
