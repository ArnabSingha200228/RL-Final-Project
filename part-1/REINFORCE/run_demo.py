#without video

import gymnasium as gym
import gymnasium_robotics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import time
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
        )
        # Reduced initial exploration noise
        self.log_std = nn.Parameter(torch.ones(act_dim) * -1.0)

    def forward(self, x):
        mean = self.net(x)
        std = self.log_std.exp()
        return Normal(mean, std)


eval_env = gym.make("FetchReach-v4", render_mode="human")
obs_dim = eval_env.observation_space["observation"].shape[0]
act_dim = eval_env.action_space.shape[0]

policy = PolicyNet(obs_dim, act_dim)

policy.load_state_dict(torch.load("reinforce_fetchreach.pt"))
policy.eval()

obs, info = eval_env.reset()

distances = []
rewards = []

for step in range(600):
    obs_tensor = torch.tensor(obs["observation"], dtype=torch.float32)
    dist_policy = policy(obs_tensor)
    action = dist_policy.mean

    obs, reward, terminated, truncated, info = eval_env.step(action.detach().numpy())

    achieved = obs["achieved_goal"]
    desired = obs["desired_goal"]
    dist = np.linalg.norm(achieved - desired)

    distances.append(dist)
    rewards.append(reward)

    print(f"{step:03d} | distance={dist:.4f} | reward={reward}")

    time.sleep(0.01)

    if terminated or truncated:
        obs, info = eval_env.reset()

eval_env.close()


# #with video

# import gymnasium as gym
# import gymnasium_robotics
# import torch
# import torch.nn as nn
# from torch.distributions import Normal
# import time
# import numpy as np
# import os

# # ===============================
# # POLICY NETWORK
# # ===============================
# class PolicyNet(nn.Module):
#     def __init__(self, obs_dim, act_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(obs_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, act_dim),
#         )
#         self.log_std = nn.Parameter(torch.ones(act_dim) * -1.0)

#     def forward(self, x):
#         mean = self.net(x)
#         std = self.log_std.exp()
#         return Normal(mean, std)

# # ===============================
# # VIDEO CONFIG
# # ===============================
# VIDEO_DIR = "videos_reinforce"
# os.makedirs(VIDEO_DIR, exist_ok=True)

# # ===============================
# # ENV WITH VIDEO RECORDING
# # ===============================
# env = gym.make(
#     "FetchReach-v4",
#     render_mode="rgb_array"
# )

# env = gym.wrappers.RecordVideo(
#     env,
#     video_folder=VIDEO_DIR,
#     episode_trigger=lambda episode_id: True,  # record EVERY episode
#     name_prefix="REINFORCE_FetchReach"
# )

# # ===============================
# # LOAD POLICY
# # ===============================
# obs_dim = env.observation_space["observation"].shape[0]
# act_dim = env.action_space.shape[0]

# policy = PolicyNet(obs_dim, act_dim)
# policy.load_state_dict(torch.load("reinforce_fetchreach.pt"))
# policy.eval()

# # ===============================
# # DEMO LOOP
# # ===============================
# obs, info = env.reset()
# episode_steps = 0
# MAX_STEPS_PER_EP = 600

# for step in range(2000):  # total demo steps
#     obs_tensor = torch.tensor(obs["observation"], dtype=torch.float32)
#     dist_policy = policy(obs_tensor)
#     action = dist_policy.mean   # deterministic demo

#     obs, reward, terminated, truncated, info = env.step(action.detach().numpy())

#     achieved = obs["achieved_goal"]
#     desired = obs["desired_goal"]
#     dist = np.linalg.norm(achieved - desired)

#     print(f"Step {step:04d} | dist={dist:.4f} | reward={reward}")

#     time.sleep(0.01)
#     episode_steps += 1

#     if terminated or truncated or episode_steps >= MAX_STEPS_PER_EP:
#         obs, info = env.reset()
#         episode_steps = 0

# env.close()
# print("Episode-wise videos saved in:", VIDEO_DIR)
