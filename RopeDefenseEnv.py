import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
import math
import random
import time

# Simulation parameters
WIDTH, HEIGHT = 1600, 1200
TRAINING_FPS = 60  # When training, run physics faster than 60 FPS.
PHYSICS_FPS = 60     # Physics simulation rate for when rendering is enabled.
GRAVITY = pygame.math.Vector2(0, 980)  # pixels/s^2
DT = 1 / PHYSICS_FPS
NUM_SEGMENTS = 3       # Number of rope segments
SEGMENT_LENGTH = 60     # Length of each rope segment in pixels
ITERATIONS = 1         # Constraint iterations for stability
DAMPING = 0.99          # Damping factor for Verlet integration
MAX_ENEMIES_OBSERVED=3

# Anchor movement constraints (circle in center)
ANCHOR_CENTER = pygame.math.Vector2(WIDTH // 2, HEIGHT // 2)
ANCHOR_RADIUS = 250    # Allowed radius for the anchor

# Ball parameters
BALL_RADIUS = 15

# Enemy parameters
ENEMY_RADIUS = 10
ENEMY_SPEED = 500  # pixels per second
ENEMY_SPAWN_INTERVAL = 500  # in milliseconds

# Explosion parameters
EXPLOSION_LIFETIME = 500  # in milliseconds
EXPLOSION_MAX_RADIUS = 40


def clamp_to_circle(point, center, radius):
    vec = point - center
    if vec.length() > radius:
        vec.scale_to_length(radius)
        return center + vec
    return point

import numpy as np

class FastRope:
    def __init__(self, anchor_pos, num_segments=NUM_SEGMENTS, segment_length=SEGMENT_LENGTH):
        self.num_segments = num_segments
        self.segment_length = segment_length
        
        # Store positions and velocities as NumPy arrays for vectorized operations
        self.positions = np.zeros((num_segments + 1, 2), dtype=np.float32)
        self.prev_positions = np.zeros((num_segments + 1, 2), dtype=np.float32)
        
        # Initialize straight down from anchor
        self.positions[:, 0] = anchor_pos[0]  # x coordinate
        self.positions[:, 1] = np.arange(num_segments + 1) * segment_length + anchor_pos[1]  # y coordinate
        self.prev_positions[:] = self.positions[:]
        
        # Pre-allocate arrays for constraint solving
        self.deltas = np.zeros((num_segments, 2), dtype=np.float32)
        self.distances = np.zeros(num_segments, dtype=np.float32)
        self.corrections = np.zeros((num_segments, 2), dtype=np.float32)
        
        # Store gravity as numpy array
        self.gravity = np.array([GRAVITY.x, GRAVITY.y], dtype=np.float32)

    def verlet(self, dt):
        # Vectorized Verlet integration
        dt2 = dt * dt
        
        # Skip first point (anchor)
        velocity = (self.positions[1:] - self.prev_positions[1:]) * DAMPING
        self.prev_positions[1:] = self.positions[1:].copy()
        self.positions[1:] += velocity + self.gravity * dt2
    
    def apply_constraints(self, anchor_pos):
        # Fix anchor point
        self.positions[0] = anchor_pos
        
        # Batch constraint solving with multiple iterations for stability
        for _ in range(ITERATIONS):
            # Calculate all segment vectors and their lengths at once
            segment_vectors = self.positions[1:] - self.positions[:-1]
            segment_lengths = np.sqrt(np.sum(segment_vectors**2, axis=1))
            
            # Avoid division by zero
            mask = segment_lengths > 1e-6
            normalized_vectors = np.zeros_like(segment_vectors)
            normalized_vectors[mask] = segment_vectors[mask] / segment_lengths[mask, np.newaxis]
            
            # Calculate correction magnitudes
            correction_magnitudes = (segment_lengths - self.segment_length) / 2.0
            
            # Apply corrections (first segment special case)
            # First segment: only move the second point
            if segment_lengths[0] > 1e-6:
                self.positions[1] -= normalized_vectors[0] * correction_magnitudes[0] * 2.0
            
            # Middle segments: distribute corrections
            for i in range(1, self.num_segments):
                if segment_lengths[i] > 1e-6:
                    correction = normalized_vectors[i] * correction_magnitudes[i]
                    self.positions[i] += correction
                    self.positions[i+1] -= correction

    def update(self, anchor_pos, dt):
        self.verlet(dt)
        self.apply_constraints(anchor_pos)

    def get_points(self):
        # Convert NumPy array back to pygame vectors for compatibility
        return [pygame.math.Vector2(self.positions[i][0], self.positions[i][1]) 
                for i in range(len(self.positions))]

    def get_ball_pos(self):
        # Return ball position as pygame vector for compatibility
        return pygame.math.Vector2(self.positions[-1][0], self.positions[-1][1])
class Rope:
    def __init__(self, anchor_pos, num_segments=NUM_SEGMENTS, segment_length=SEGMENT_LENGTH):
        self.num_segments = num_segments
        self.segment_length = segment_length
        self.points = []
        for i in range(num_segments + 1):
            pos = pygame.math.Vector2(anchor_pos[0], anchor_pos[1] + i * segment_length)
            self.points.append({
                'pos': pos.copy(),
                'prev_pos': pos.copy()
            })

    def verlet(self, dt):
        # Use squared dt for gravity term
        dt2 = dt * dt
        for i in range(1, len(self.points)):
            point = self.points[i]
            current = point['pos']
            prev = point['prev_pos']
            # Compute velocity using Verlet integration with damping
            velocity = (current - prev) * DAMPING
            point['prev_pos'] = current.copy()
            point['pos'] = current + velocity + GRAVITY * dt2

    def apply_constraints(self, anchor_pos):
        # Fix the first point at the anchor
        self.points[0]['pos'] = pygame.math.Vector2(anchor_pos)
        for _ in range(ITERATIONS):
            for i in range(len(self.points) - 1):
                p1 = self.points[i]['pos']
                p2 = self.points[i+1]['pos']
                delta = p2 - p1
                dist = delta.length()
                if dist == 0:
                    continue
                diff = (dist - self.segment_length) / dist
                # For the first segment, only adjust the second point
                if i == 0:
                    p2 -= delta * diff
                else:
                    p1 += delta * (diff * 0.5)
                    p2 -= delta * (diff * 0.5)
                self.points[i]['pos'] = p1
                self.points[i+1]['pos'] = p2

    def update(self, anchor_pos, dt):
        self.verlet(dt)
        self.apply_constraints(anchor_pos)

    def get_points(self):
        return [p['pos'] for p in self.points]

    def get_ball_pos(self):
        # The ball is at the last point.
        return self.points[-1]['pos']


class Enemy:
    def __init__(self, pos=None):
        # Spawn enemy at a random edge of the screen
        side = random.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top':
            x = random.uniform(0, WIDTH)
            y = -ENEMY_RADIUS
            direction = pygame.math.Vector2(random.uniform(-1, 1), 1)
        elif side == 'bottom':
            x = random.uniform(0, WIDTH)
            y = HEIGHT + ENEMY_RADIUS
            direction = pygame.math.Vector2(random.uniform(-1, 1), -1)
        elif side == 'left':
            x = -ENEMY_RADIUS
            y = random.uniform(0, HEIGHT)
            direction = pygame.math.Vector2(1, random.uniform(-1, 1))
        else:  # right
            x = WIDTH + ENEMY_RADIUS
            y = random.uniform(0, HEIGHT)
            direction = pygame.math.Vector2(-1, random.uniform(-1, 1))
        # randomly direct the enemy towards the current anchor position
        directToPlayer = random.choice([True, False])
        if directToPlayer:
            direction = pygame.math.Vector2(pos.x - x, pos.y - y)
        self.pos = pygame.math.Vector2(x, y)
        self.vel = direction.normalize() * ENEMY_SPEED

    def update(self, dt):
        self.pos += self.vel * dt

    def check_collision(self, ball_pos):
        return (self.pos - ball_pos).length() < (ENEMY_RADIUS + BALL_RADIUS)

    def check_anchor_collision(self, anchor):
        # Collision with a small circle around the anchor
        return (self.pos - anchor).length() < (ENEMY_RADIUS + 5)


class Explosion:
    def __init__(self, pos, start_time):
        self.pos = pygame.math.Vector2(pos)
        self.start_time = start_time
        self.lifetime = EXPLOSION_LIFETIME

    def is_active(self, current_time):
        return (current_time - self.start_time) <= self.lifetime

    def draw(self, screen, current_time):
        elapsed = current_time - self.start_time
        progress = elapsed / self.lifetime
        radius = progress * EXPLOSION_MAX_RADIUS
        alpha = 255 * (1 - progress)
        surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surface, (255, 100, 0, int(alpha)), (int(radius), int(radius)), int(radius))
        screen.blit(surface, (self.pos.x - radius, self.pos.y - radius))


class RopeEnv(gym.Env):
    """
    Gym environment for the rope simulation.
    The agent controls the anchor position (subject to clamping) by providing a 2D continuous action.
    Enhanced observation space provides rich state representation for better agent learning.
    """
    metadata = {"render_modes": ["human"], "render_fps": PHYSICS_FPS}

    def __init__(self, render_mode=None, simulation_fps=TRAINING_FPS, max_enemies_observed=MAX_ENEMIES_OBSERVED):
        super().__init__()
        self.render_mode = render_mode
        self.simulation_fps = simulation_fps
        self.dt = 1 / simulation_fps
        self.max_enemies_observed = max_enemies_observed  # Number of closest enemies to include in observation

        # Define action and observation space
        # Action: 2D continuous change in anchor position (dx, dy)
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float32)

        # Enhanced observation space: Include angular velocities and energy metrics
        numAngles = NUM_SEGMENTS
        self.obs_dim = (
            2 + 2 + 2 + 2 + numAngles + numAngles + 9 * self.max_enemies_observed + 2
        )  # Added angular velocities and energy metrics
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Initialize simulation elements
        self.anchor = ANCHOR_CENTER.copy()
        self.rope = Rope(self.anchor)
        self.enemies = []
        self.explosions = []
        self.score = 0
        self.game_over = False

        # Store previous ball position to calculate velocity
        self.prev_ball_pos = self.rope.get_ball_pos()
        self.ball_velocity = pygame.math.Vector2(0, 0)
        self.anchor_velocity = pygame.math.Vector2(0, 0)
        self.rope_angles = []
        self.rope_angular_velocities = []  # New feature

        self.enemy_timer = 0
        self.enemy_spawn_interval = ENEMY_SPAWN_INTERVAL

        # Additional performance metrics
        self.episode_steps = 0
        self.total_reward = 0
        self.kinetic_energy = 0  # New feature
        self.potential_energy = 0  # New feature

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("RopeEnv")
            self.clock = pygame.time.Clock()

    def step(self, action):
        self.episode_steps += 1

        # Apply action to update the anchor.
        old_anchor = self.anchor.copy()
        new_anchor = self.anchor + pygame.math.Vector2(action[0], action[1])
        # Clamp anchor to allowed circle.
        self.anchor = clamp_to_circle(new_anchor, ANCHOR_CENTER, ANCHOR_RADIUS)

        # Store previous ball position for velocity calculation
        self.prev_ball_pos = self.rope.get_ball_pos().copy()

        # Update rope simulation
        self.rope.update(self.anchor, self.dt)
        ball_pos = self.rope.get_ball_pos()

        # Calculate ball velocity
        self.ball_velocity = (ball_pos - self.prev_ball_pos) / self.dt
        self.anchor_velocity = (self.anchor - old_anchor) / self.dt

        # Precompute rope angles and angular velocities
        self.rope_angles = []
        self.rope_angular_velocities = []
        points = self.rope.get_points()
        for i in range(len(points) - 1):
            segment = points[i + 1] - points[i]
            angle = math.atan2(segment.y, segment.x)
            self.rope_angles.append(angle)

            # Calculate angular velocity
            if len(self.rope_angles) > 1:
                angular_velocity = (self.rope_angles[-1] - self.rope_angles[-2]) / self.dt
                self.rope_angular_velocities.append(angular_velocity)

        # Calculate energy metrics
        self.kinetic_energy = sum(
            0.5 * (point - prev_point).length_squared() for point, prev_point in zip(points, self.rope.get_points())
        )
        self.potential_energy = sum(
            GRAVITY.y * (point.y - ANCHOR_CENTER.y) for point in points
        )

        current_time = pygame.time.get_ticks() if self.render_mode else self.enemy_timer

        # Update enemy spawn timer
        self.enemy_timer += self.dt * 1000  # Convert to milliseconds

        # Spawn new enemy at intervals (only if not game over)
        if not self.game_over and (self.enemy_timer >= self.enemy_spawn_interval):
            self.enemy_timer = 0
            self.enemies.append(Enemy(self.anchor))

        # Update enemies and check collisions
        enemies_to_remove = []
        for enemy in self.enemies:
            enemy.update(self.dt)
            if enemy.check_collision(ball_pos):
                enemies_to_remove.append(enemy)
                self.score += 1
                if self.render_mode:
                    self.explosions.append(Explosion(enemy.pos, current_time))
            if enemy.check_anchor_collision(self.anchor):
                self.game_over = True

        for enemy in enemies_to_remove:
            if enemy in self.enemies:
                self.enemies.remove(enemy)

        # Clean up off-screen enemies
        self.enemies = [
            e
            for e in self.enemies
            if -50 < e.pos.x < WIDTH + 50 and -50 < e.pos.y < HEIGHT + 50
        ]

        if self.render_mode:
            # Clean up explosions that have expired
            self.explosions = [
                exp for exp in self.explosions if exp.is_active(current_time)
            ]

        # Calculate observation
        obs = self._get_observation(ball_pos)

        # Reward calculation:
        reward = 10.0 * len(enemies_to_remove)  # +10 for each enemy hit

        # # Penalize effort (quadratic penalty)
        # reward -= np.square(action).sum() * 0.1

        # # Reward staying alive
        # reward += 0.1

        # # Reward staying close to the center (exponential decay)
        # reward += np.exp(-0.01 * (self.anchor - ANCHOR_CENTER).length())

        # # Game over penalty
        # if self.game_over:
        #     reward -= 100.0

        self.total_reward += reward
        terminated = self.game_over
        truncated = self.episode_steps >= 10000
        info = {
            "score": self.score,
            "enemy_count": len(self.enemies),
            "episode_steps": self.episode_steps,
            "total_reward": self.total_reward,
        }

        # If rendering is enabled, draw the current frame.
        if self.render_mode:
            self._render_frame()

        return obs, reward, terminated, truncated, info

    def _get_observation(self, ball_pos):
        """
        Construct the enhanced observation vector.
        """
        obs = []
        obs.extend([self.anchor.x, self.anchor.y])
        obs.extend([self.anchor_velocity.x, self.anchor_velocity.y])
        obs.extend([ball_pos.x, ball_pos.y])
        obs.extend([self.ball_velocity.x, self.ball_velocity.y])

        # Add rope angles and angular velocities
        max_segments = NUM_SEGMENTS  # example limit
        angles = self.rope_angles[:max_segments] + [0.0] * (
            max_segments - len(self.rope_angles[:max_segments])
        )
        angular_velocities = self.rope_angular_velocities[:max_segments] + [
            0.0
        ] * (max_segments - len(self.rope_angular_velocities[:max_segments]))
        obs.extend(angles)
        obs.extend(angular_velocities)

        # Sort enemies by distance and include positions or polar coords
        sorted_enemies = sorted(
            self.enemies, key=lambda e: (e.pos - ball_pos).length()
        )[: self.max_enemies_observed]
        for enemy in sorted_enemies:
            offset = enemy.pos - ball_pos
            obs.extend([offset.x, offset.y])
            # also add coords wrt anchor
            offset = enemy.pos - self.anchor
            obs.extend([offset.x, offset.y])

            # show relative velocity
            obs.extend([enemy.vel.x, enemy.vel.y])
            # add features: distance, angle, velocity
            obs.extend(
                [
                    offset.length(),
                    math.atan2(offset.y, offset.x),
                    enemy.vel.length(),
                ]
            )

        # Pad if fewer enemies
        for _ in range(self.max_enemies_observed - len(sorted_enemies)):
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Add energy metrics
        obs.extend([self.kinetic_energy, self.potential_energy])

        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.anchor = ANCHOR_CENTER.copy()
        self.rope = Rope(self.anchor)
        self.enemies = []
        self.explosions = []
        self.score = 0
        self.game_over = False
        
        self.prev_ball_pos = self.rope.get_ball_pos()
        self.ball_velocity = pygame.math.Vector2(0, 0)
        self.anchor_velocity = pygame.math.Vector2(0, 0)
        self.rope_angles = []
        self.rope_angular_velocities = []  # New feature
        
        self.enemy_timer = 0
        self.episode_steps = 0
        self.total_reward = 0
        self.kinetic_energy = 0  # New feature
        self.potential_energy = 0  # New feature
        
        # Return initial observation
        ball_pos = self.rope.get_ball_pos()
        observation = self._get_observation(ball_pos)
        info = {}
        return observation, info

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You may experience unexpected behavior. "
                "The available render modes are human"
            )
            return
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        # Process pygame events to allow window closing.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((30, 30, 30))
        # Draw allowed anchor circle
        pygame.draw.circle(self.screen, (50, 50, 50), (int(ANCHOR_CENTER.x), int(ANCHOR_CENTER.y)), ANCHOR_RADIUS, 2)
        # Draw rope segments
        points = self.rope.get_points()
        for i in range(len(points) - 1):
            pygame.draw.line(self.screen, (200, 200, 200),
                             (int(points[i].x), int(points[i].y)),
                             (int(points[i+1].x), int(points[i+1].y)), 2)
        # Draw ball
        ball_pos = self.rope.get_ball_pos()
        pygame.draw.circle(self.screen, (255, 100, 100), (int(ball_pos.x), int(ball_pos.y)), BALL_RADIUS)
        # Draw anchor
        pygame.draw.circle(self.screen, (100, 255, 100), (int(self.anchor.x), int(self.anchor.y)), 5)
        # Draw enemies
        for enemy in self.enemies:
            pygame.draw.circle(self.screen, (255, 255, 0), (int(enemy.pos.x), int(enemy.pos.y)), ENEMY_RADIUS)
        # Draw explosions
        current_time = pygame.time.get_ticks()
        for exp in self.explosions:
            exp.draw(self.screen, current_time)
        # Draw score
        font = pygame.font.SysFont("Arial", 24)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        pygame.display.flip()
        self.clock.tick(self.simulation_fps)

    def close(self):
        if self.render_mode:
            pygame.quit()


def benchmark_env(env, steps=1000):
    """
    Run the environment without rendering for a given number of steps and print the effective steps per second.
    """
    start = time.time()
    obs, _ = env.reset()
    for i in range(steps):
        # Sample a random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    end = time.time()
    total_time = end - start
    fps = steps / total_time
    print(f"Benchmark: {steps} steps in {total_time:.2f} seconds, FPS = {fps:.2f}")


if __name__ == '__main__':
    # Run benchmark in training mode (no rendering)
    print("Running benchmark in training mode (no rendering)...")
    env = RopeEnv(render_mode=False, simulation_fps=TRAINING_FPS)
    benchmark_env(env, steps=10000)
    env.close()

    # Optionally, run a demo with rendering (press close window to quit)
    demo = input("Run demo with rendering? (y/n): ")
    if demo.lower().startswith("y"):
        env = RopeEnv(render_mode="human", simulation_fps=PHYSICS_FPS)
        obs, _ = env.reset()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # For demo, use mouse position difference as action
            mouse = pygame.mouse.get_pos()
            anchor = np.array([env.anchor.x, env.anchor.y])
            action = np.array(mouse) - anchor
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()