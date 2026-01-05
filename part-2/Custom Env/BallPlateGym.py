import pygame
import math
import sys
import numpy as np

import gymnasium as gym
from gymnasium import spaces

# --- CONSTANTS ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
KEY_TILT_STEP = 0.8   # degrees per frame from keyboard
GRAVITY = 0.15

# --- COLORS ---
SKY_BLUE = (135, 206, 235)
GROUND_BROWN = (139, 111, 78)
PLATE_TOP = (200, 200, 210)
PLATE_EDGE = (80, 80, 90)
BALL_RED = (200, 40, 40)
BALL_SHADOW_COLOR = (120, 20, 20)
TABLE_BROWN = (101, 67, 33)
TABLE_LIGHT = (139, 90, 45)
TABLE_DARK = (80, 53, 26)
BLACK = (0, 0, 0)


# --- HELPER CLASSES (Physics and Logic) ---
class Ball:
    def __init__(self):
        self.radius = 10
        self.reset()

    def update(self, plate):
        is_horizontally_on_plate = (abs(self.x) <= plate.width / 2 - self.radius and
                                    abs(self.z) <= plate.depth / 2 - self.radius)

        if self.on_plate and is_horizontally_on_plate:
            accel_x = -GRAVITY * math.sin(math.radians(plate.tilt_z))
            accel_z = -GRAVITY * math.sin(math.radians(plate.tilt_x))
            self.vx += accel_x
            self.vz += accel_z
            surface_y = plate.get_surface_height_at(self.x, self.z)
            self.y = surface_y + self.radius
            self.vx *= 0.99
            self.vz *= 0.99
            self.vy = 0
        else:
            self.on_plate = False
            self.vy -= 0.4

        self.x += self.vx
        self.y += self.vy
        self.z += self.vz

        if self.y <= self.radius:
            self.y = self.radius
            self.vy = -self.vy * 0.6
        
        return self.on_plate

    def reset(self, initial_pos=None):
        # --- FIX: Check for None explicitly to avoid the ValueError ---
        if initial_pos is not None:
            self.x, self.z = initial_pos
        else:
            self.x, self.z = 0.0, 0.0
            
        self.y = 30.0
        self.vx, self.vy, self.vz = 0.0, 0.0, 0.0
        self.on_plate = True


class Plate:
    def __init__(self):
        self.width = 180
        self.depth = 140
        self.base_height = 60
        self.max_tilt = 25.0
        self.reset()

    def update(self, action):
        tilt_change_x, tilt_change_z = action
        self.tilt_x += tilt_change_x
        self.tilt_z += tilt_change_z
        self.tilt_x = np.clip(self.tilt_x, -self.max_tilt, self.max_tilt)
        self.tilt_z = np.clip(self.tilt_z, -self.max_tilt, self.max_tilt)

    def reset(self, initial_tilt=None):
        # --- FIX: Check for None explicitly to avoid the ValueError ---
        if initial_tilt is not None:
            self.tilt_x, self.tilt_z = initial_tilt
        else:
            self.tilt_x, self.tilt_z = 0.0, 0.0

    def get_surface_height_at(self, x_pos, z_pos):
        tilt_x_rad = math.radians(-self.tilt_x)
        tilt_z_rad = math.radians(self.tilt_z)
        y_offset_from_x_tilt = -z_pos * math.sin(tilt_x_rad)
        y_offset_from_z_tilt = x_pos * math.sin(tilt_z_rad)
        return self.base_height + y_offset_from_x_tilt * math.cos(tilt_z_rad) + y_offset_from_z_tilt


# --- GYMNASIUM ENVIRONMENT CLASS ---
class BallBalancerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, render_mode=None, max_steps=1200):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.screen = None
        self.clock = None
        self.step_count = 0
        
        self.plate = Plate()
        self.ball = Ball()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)


    def _get_keyboard_action(self):
        """
        Read arrow keys and convert to plate tilt deltas.
        This allows external human interruption.
        """
        keys = pygame.key.get_pressed()
        tilt_x = 0.0
        tilt_z = 0.0

        if keys[pygame.K_UP]:
            tilt_x -= KEY_TILT_STEP
        if keys[pygame.K_DOWN]:
            tilt_x += KEY_TILT_STEP
        if keys[pygame.K_LEFT]:
            tilt_z += KEY_TILT_STEP
        if keys[pygame.K_RIGHT]:
            tilt_z -= KEY_TILT_STEP

        return np.array([tilt_x, tilt_z], dtype=np.float32)

    def _get_obs(self):
        return np.array([
            self.ball.x / (self.plate.width / 2),
            self.ball.z / (self.plate.depth / 2),
            self.ball.vx,
            self.ball.vz,
            self.plate.tilt_x / self.plate.max_tilt,
            self.plate.tilt_z / self.plate.max_tilt
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        initial_tilt = self.np_random.uniform(low=-self.plate.max_tilt / 2, high=self.plate.max_tilt / 2, size=(2,))
        self.plate.reset(initial_tilt=initial_tilt)
        
        max_x = self.plate.width / 2 - self.ball.radius
        max_z = self.plate.depth / 2 - self.ball.radius
        initial_pos = self.np_random.uniform(low=(-max_x, -max_z), high=(max_x, max_z), size=(2,))
        self.ball.reset(initial_pos=initial_pos)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {}

    def step(self, action):
        # --- External keyboard intervention ---
        human_action = np.zeros(2, dtype=np.float32)

        if self.render_mode == "human":
            human_action = self._get_keyboard_action()

        # Combine agent action + human input
        combined_action = action + human_action

        # Clip to safe limits
        combined_action = np.clip(combined_action, -2.0, 2.0)

        self.plate.update(combined_action)

        ball_on_plate = self.ball.update(self.plate)
        
        terminated = not ball_on_plate
        
        if terminated:
            reward = -100.0
        else:
            dist_from_center = np.sqrt((self.ball.x / (self.plate.width / 2))**2 + (self.ball.z / (self.plate.depth / 2))**2)
            reward = 1.0 - dist_from_center

        observation = self._get_obs()
        
        if self.render_mode == "human":
            self._render_frame()

        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        
        return observation, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Gymnasium Ball Balancer")
            self.clock = pygame.time.Clock()
        
        self.screen.fill(SKY_BLUE)
        self._draw_environment()
        self._draw_table_base()
        self._draw_plate()
        self._draw_ball()
        font = pygame.font.SysFont(None, 24)
        txt = font.render(
            f"Tilt X: {self.plate.tilt_x:.1f} | Tilt Z: {self.plate.tilt_z:.1f}",
            True, BLACK
        )
        self.screen.blit(txt, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    # --- Drawing Helper Methods ---
    def _isometric_projection(self, x, y, z):
        angle_x = math.radians(30)
        angle_y = math.radians(-45)
        iso_x = x * math.cos(angle_y) - z * math.sin(angle_y)
        iso_y = x * math.sin(angle_y) * math.sin(angle_x) + y * math.cos(angle_x) + z * math.cos(angle_y) * math.sin(angle_x)
        scale = 2.0
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT - 150
        return int(center_x + iso_x * scale), int(center_y - iso_y * scale)

    def _draw_environment(self):
        ground_points = [
            self._isometric_projection(-300, 0, -300), self._isometric_projection(300, 0, -300),
            self._isometric_projection(300, 0, 300), self._isometric_projection(-300, 0, 300)
        ]
        pygame.draw.polygon(self.screen, GROUND_BROWN, ground_points)

    def _draw_table_base(self):
        w, d, h = self.plate.width + 20, self.plate.depth + 20, self.plate.base_height
        corners = [(-w/2, h, -d/2), (w/2, h, -d/2), (w/2, h, d/2), (-w/2, h, d/2),
                   (-w/2, 0, -d/2), (w/2, 0, -d/2), (w/2, 0, d/2), (-w/2, 0, d/2)]
        sc = [self._isometric_projection(x, y, z) for x, y, z in corners]
        faces = [([sc[3], sc[2], sc[6], sc[7]], TABLE_DARK), ([sc[0], sc[1], sc[5], sc[4]], TABLE_BROWN), 
                 ([sc[0], sc[1], sc[2], sc[3]], TABLE_LIGHT)]
        for points, color in faces:
            pygame.draw.polygon(self.screen, color, points)
            pygame.draw.polygon(self.screen, TABLE_DARK, points, 2)

    def _draw_plate(self):
        corners_3d = self._get_plate_corners()
        corners_2d = [self._isometric_projection(x, y, z) for x, y, z in corners_3d]
        pygame.draw.polygon(self.screen, PLATE_TOP, corners_2d)
        pygame.draw.polygon(self.screen, PLATE_EDGE, corners_2d, 2)

    def _get_plate_corners(self):
        w, d, h = self.plate.width / 2, self.plate.depth / 2, self.plate.base_height
        corners = [(-w, h, -d), (w, h, -d), (w, h, d), (-w, h, d)]
        tx_rad, tz_rad = math.radians(-self.plate.tilt_x), math.radians(self.plate.tilt_z)
        transformed = []
        for x, y, z in corners:
            y_rel = y - h
            y_rot_x = y_rel * math.cos(tx_rad) - z * math.sin(tx_rad)
            z_rot = y_rel * math.sin(tx_rad) + z * math.cos(tx_rad)
            x_rot = x * math.cos(tz_rad) - y_rot_x * math.sin(tz_rad)
            y_rot_z = x * math.sin(tz_rad) + y_rot_x * math.cos(tz_rad)
            transformed.append((x_rot, y_rot_z + h, z_rot))
        return transformed

    def _draw_ball(self):
        shadow_y_3d = 0
        if self.ball.on_plate:
            shadow_y_3d = self.plate.get_surface_height_at(self.ball.x, self.ball.z)
        shadow_x, shadow_y = self._isometric_projection(self.ball.x, shadow_y_3d, self.ball.z)
        height = self.ball.y - shadow_y_3d
        shadow_size = max(3, self.ball.radius - int(height / 6))
        shadow_alpha = max(20, 120 - int(height * 1.5))
        if shadow_size > 0:
            shadow_surf = pygame.Surface((shadow_size * 2, shadow_size * 2), pygame.SRCALPHA)
            pygame.draw.ellipse(shadow_surf, (0, 0, 0, shadow_alpha), shadow_surf.get_rect())
            self.screen.blit(shadow_surf, shadow_surf.get_rect(center=(shadow_x, shadow_y)))
        
        ball_x, ball_y = self._isometric_projection(self.ball.x, self.ball.y, self.ball.z)
        pygame.draw.circle(self.screen, BALL_RED, (ball_x, ball_y), self.ball.radius)
        pygame.draw.circle(self.screen, BALL_SHADOW_COLOR, (ball_x, ball_y), self.ball.radius, 1)


# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    env = BallBalancerEnv(render_mode="human")
    
    print("Action Space:", env.action_space)
    print("Observation Space:", env.observation_space)

    episodes = 5
    for episode in range(episodes):
        obs, info = env.reset()
        terminated = False
        total_reward = 0
        step_count = 0
        while not terminated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
        
        print(f"Episode {episode + 1}: Finished after {step_count} steps. Total Reward: {total_reward:.2f}")

    env.close()