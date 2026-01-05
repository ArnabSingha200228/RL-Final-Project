from gymnasium.envs.registration import register

# Normal (manual physics) environment
register(
    id="BallPlate-v0",
    entry_point="BallPlateGym:BallBalancerEnv",
    max_episode_steps=1200,
)

# Box2D environment
register(
    id="BallPlateBox2D-v0",
    entry_point="BallPlateGymBox2d:BallBalancerEnv",
    max_episode_steps=1200,
)