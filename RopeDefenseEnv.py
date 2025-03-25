import gym
from gym import spaces
import numpy as np
import pygame
import sys
import math
import random
import time

# Simulation parameters
WIDTH, HEIGHT = 800, 600
TRAINING_FPS = 60  # When training, run physics faster than 60 FPS.
PHYSICS_FPS = 60     # Physics simulation rate for when rendering is enabled.
GRAVITY = pygame.math.Vector2(0, 980)  # pixels/s^2
DT = 1 / PHYSICS_FPS
NUM_SEGMENTS = 20       # Number of rope segments
SEGMENT_LENGTH = 20     # Length of each rope segment in pixels
ITERATIONS = 10         # Constraint iterations for stability
DAMPING = 0.99          # Damping factor for Verlet integration

# Anchor movement constraints (circle in center)
ANCHOR_CENTER = pygame.math.Vector2(WIDTH // 2, HEIGHT // 2)
ANCHOR_RADIUS = 250    # Allowed radius for the anchor

# Ball parameters
BALL_RADIUS = 15

# Enemy parameters
ENEMY_RADIUS = 10
ENEMY_SPEED = 1000  # pixels per second
ENEMY_SPAWN_INTERVAL = 200  # in milliseconds

# Explosion parameters
EXPLOSION_LIFETIME = 500  # in milliseconds
EXPLOSION_MAX_RADIUS = 40


def clamp_to_circle(point, center, radius):
    vec = point - center
    if vec.length() > radius:
        vec.scale_to_length(radius)
        return center + vec
    return point


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
    metadata = {"render.modes": ["human"]}

    def __init__(self, render_mode=False, simulation_fps=TRAINING_FPS, max_enemies_observed=5):
        super(RopeEnv, self).__init__()
        self.render_mode = render_mode
        self.simulation_fps = simulation_fps
        self.dt = 1 / simulation_fps
        self.max_enemies_observed = max_enemies_observed  # Number of closest enemies to include in observation

        # Define action and observation space
        # Action: 2D continuous change in anchor position (dx, dy)
        self.action_space = spaces.Box(low=-50.0, high=50.0, shape=(2,), dtype=np.float32)
        
        # Enhanced observation space:
        # [anchor_x, anchor_y,                     # Anchor position (2)
        #  ball_x, ball_y,                         # Ball position (2)
        #  ball_vx, ball_vy,                       # Ball velocity (2)
        #  enemy1_rel_x, enemy1_rel_y,             # Relative positions of up to 5 enemies (10)
        #  enemy2_rel_x, enemy2_rel_y,
        #  ...
        #  enemy5_rel_x, enemy5_rel_y,
        #  enemy1_rel_angle, enemy1_ball_vel_angle, # Enemy angles (10)
        #  enemy2_rel_angle, enemy2_ball_vel_angle,
        #  ...
        #  enemy5_rel_angle, enemy5_ball_vel_angle,
        #  enemy_spawn_timer_normalized,           # Normalized spawn timer (1)
        #  score]                                  # Score (1)
        
        # Total observation size: 2 + 2 + 2 + (2*max_enemies) + (2*max_enemies) + 1 + 1 = 8 + 4*max_enemies
        obs_size = 8 + 4 * max_enemies_observed
        
        # Set reasonable bounds for each observation component
        low_values = []
        high_values = []
        
        # Anchor position (normalized to [0,1])
        low_values.extend([0, 0])
        high_values.extend([1, 1])
        
        # Ball position (normalized to [0,1])
        low_values.extend([0, 0])
        high_values.extend([1, 1])
        
        # Ball velocity (normalized to [-1,1])
        low_values.extend([-1, -1])
        high_values.extend([1, 1])
        
        # Enemy relative positions (normalized to [-1,1])
        for _ in range(max_enemies_observed):
            low_values.extend([-1, -1])
            high_values.extend([1, 1])
        
        # Enemy angles (normalized to [-1,1])
        for _ in range(max_enemies_observed * 2):
            low_values.extend([-1])
            high_values.extend([1])
        
        # Spawn timer (normalized to [0,1])
        low_values.append(0)
        high_values.append(1)
        
        # Score (unbounded)
        low_values.append(0)
        high_values.append(1e5)
        
        self.observation_space = spaces.Box(
            low=np.array(low_values, dtype=np.float32),
            high=np.array(high_values, dtype=np.float32),
            dtype=np.float32
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

        self.enemy_timer = 0
        self.enemy_spawn_interval = ENEMY_SPAWN_INTERVAL

        # Additional performance metrics
        self.episode_steps = 0
        self.total_reward = 0
        
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("RopeEnv")
            self.clock = pygame.time.Clock()

    def step(self, action):
        self.episode_steps += 1
        
        # Apply action to update the anchor.
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
        self.enemies = [e for e in self.enemies if -50 < e.pos.x < WIDTH + 50 and -50 < e.pos.y < HEIGHT + 50]

        if self.render_mode:
            # Clean up explosions that have expired
            self.explosions = [exp for exp in self.explosions if exp.is_active(current_time)]

        # Calculate observation
        obs = self._get_observation(ball_pos)

        # Reward calculation:
        # +1 for each enemy hit
        reward = 1.0 * len(enemies_to_remove)

        # penalize effort
        reward -= np.abs(action).sum() * 0.01
        
        # Game over penalty
        if self.game_over:
            reward -= 10.0
        
        self.total_reward += reward
        done = self.game_over
        
        # Add max episode length termination condition
        if self.episode_steps >= 10000:
            done = True
        
        info = {
            'score': self.score,
            'enemy_count': len(self.enemies),
            'episode_steps': self.episode_steps,
            'total_reward': self.total_reward,
        }

        # If rendering is enabled, draw the current frame.
        if self.render_mode:
            self._render_frame()

        return obs, reward, done, info

    def _get_observation(self, ball_pos):
        """
        Construct the enhanced observation vector.
        """
        # Normalize positions to [0,1] range
        norm_anchor = pygame.math.Vector2(self.anchor.x / WIDTH, self.anchor.y / HEIGHT)
        norm_ball_pos = pygame.math.Vector2(ball_pos.x / WIDTH, ball_pos.y / HEIGHT)
        
        # Normalize velocity to [-1,1] range by dividing by a reasonable maximum velocity
        # Using 1000 pixels/s as a normalization factor
        max_vel = 1000.0
        norm_ball_vel = pygame.math.Vector2(
            np.clip(self.ball_velocity.x / max_vel, -1.0, 1.0),
            np.clip(self.ball_velocity.y / max_vel, -1.0, 1.0)
        )
        
        # Start building observation vector
        obs = [
            norm_anchor.x, norm_anchor.y,
            norm_ball_pos.x, norm_ball_pos.y,
            norm_ball_vel.x, norm_ball_vel.y,
        ]
        
        # Get the up to max_enemies_observed closest enemies
        enemy_features = []
        if self.enemies:
            # Calculate distance from ball for each enemy
            enemies_with_dist = [(enemy, (enemy.pos - ball_pos).length()) for enemy in self.enemies]
            # Sort by distance
            enemies_with_dist.sort(key=lambda x: x[1])
            # Take the closest ones
            closest_enemies = enemies_with_dist[:self.max_enemies_observed]
            
            for enemy, dist in closest_enemies:
                # Relative position to ball (normalized to [-1,1])
                rel_pos = enemy.pos - ball_pos
                norm_rel_pos = pygame.math.Vector2(
                    np.clip(rel_pos.x / WIDTH, -1.0, 1.0),
                    np.clip(rel_pos.y / HEIGHT, -1.0, 1.0)
                )
                
                # Relative angle between ball-enemy vector and anchor-ball vector
                anchor_ball_vec = ball_pos - self.anchor
                if anchor_ball_vec.length() > 0:
                    anchor_ball_vec.normalize_ip()
                    
                enemy_vec = enemy.pos - ball_pos
                if enemy_vec.length() > 0:
                    enemy_vec.normalize_ip()
                    
                # Calculate dot product and convert to angle
                dot_product = max(min(anchor_ball_vec.dot(enemy_vec), 1.0), -1.0)
                # rel_angle = math.acos(dot_product) / math.pi  # Normalize to [0,1]
                cross = anchor_ball_vec.cross(enemy_vec)
                angle = math.atan2(cross, dot_product)
                rel_angle = angle / math.pi  # Normalized to [-1, 1]
                
                # Angle between ball velocity and enemy direction
                if self.ball_velocity.length() > 0 and enemy_vec.length() > 0:
                    ball_vel_norm = self.ball_velocity.normalize()
                    dot_product_vel = max(min(ball_vel_norm.dot(enemy_vec), 1.0), -1.0)
                    # vel_angle = math.acos(dot_product_vel) / math.pi  # Normalize to [0,1]
                    cross = ball_vel_norm.cross(enemy_vec)
                    angle = math.atan2(cross, dot_product_vel)
                    vel_angle = angle / math.pi # Normalized to [-1, 1]
                else:
                    vel_angle = 0  # Default when no velocity or direction
                
                # Add these features to our list
                enemy_features.extend([norm_rel_pos.x, norm_rel_pos.y, rel_angle, vel_angle])
        
        # Pad with zeros for missing enemies
        while len(enemy_features) < 4 * self.max_enemies_observed:
            enemy_features.extend([0.0, 0.0, 0.0, 0.0])  # Add dummy enemy features
            
        # Add enemy features to observation
        obs.extend(enemy_features)
        
        # Add normalized enemy spawn timer
        obs.append(self.enemy_timer / ENEMY_SPAWN_INTERVAL)
        
        # Add score
        obs.append(float(self.score))
        
        return np.array(obs, dtype=np.float32)

    def reset(self):
        self.anchor = ANCHOR_CENTER.copy()
        self.rope = Rope(self.anchor)
        self.enemies = []
        self.explosions = []
        self.score = 0
        self.game_over = False
        
        self.prev_ball_pos = self.rope.get_ball_pos()
        self.ball_velocity = pygame.math.Vector2(0, 0)
        
        self.enemy_timer = 0
        self.episode_steps = 0
        self.total_reward = 0
        
        # Return initial observation
        ball_pos = self.rope.get_ball_pos()
        return self._get_observation(ball_pos)

    def render(self, mode="human"):
        if not self.render_mode:
            return
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
    obs = env.reset()
    for i in range(steps):
        # Sample a random action
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        if done:
            obs = env.reset()
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
        env = RopeEnv(render_mode=True, simulation_fps=PHYSICS_FPS)
        obs = env.reset()
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
            obs, reward, done, _ = env.step(action)
            if done:
                obs = env.reset()
        env.close()


