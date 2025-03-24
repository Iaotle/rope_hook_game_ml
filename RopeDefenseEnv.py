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
TRAINING_FPS = 1000  # When training, run physics faster than 60 FPS.
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
ENEMY_RADIUS = 15
ENEMY_SPEED = 200  # pixels per second
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
    def __init__(self):
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
    The observation consists of:
      [anchor_x, anchor_y, ball_x, ball_y, rel_enemy_x, rel_enemy_y, score]
    If no enemy is present, the enemy-related values are zeros.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, render_mode=False, simulation_fps=PHYSICS_FPS):
        super(RopeEnv, self).__init__()
        self.render_mode = render_mode
        self.simulation_fps = simulation_fps
        self.dt = 1 / simulation_fps

        # Define action and observation space
        # Action: 2D continuous change in anchor position (dx, dy)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
        # Observation: [anchor_x, anchor_y, ball_x, ball_y, enemy_rel_x, enemy_rel_y, score]
        # Coordinates in pixels and score as a float.
        obs_low = np.array([0, 0, 0, 0, -WIDTH, -HEIGHT, 0], dtype=np.float32)
        obs_high = np.array([WIDTH, HEIGHT, WIDTH, HEIGHT, WIDTH, HEIGHT, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Initialize simulation elements
        self.anchor = ANCHOR_CENTER.copy()
        self.rope = Rope(self.anchor)
        self.enemies = []
        self.explosions = []
        self.score = 0
        self.game_over = False

        self.enemy_timer = pygame.time.get_ticks()

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("RopeEnv")
            self.clock = pygame.time.Clock()

    def step(self, action):
        # Apply action to update the anchor.
        # For simplicity, the action is added to the current anchor position.
        new_anchor = self.anchor + pygame.math.Vector2(action[0], action[1])
        # Clamp anchor to allowed circle.
        self.anchor = clamp_to_circle(new_anchor, ANCHOR_CENTER, ANCHOR_RADIUS)

        # Update rope simulation
        self.rope.update(self.anchor, self.dt)
        ball_pos = self.rope.get_ball_pos()

        current_time = pygame.time.get_ticks()

        # Spawn new enemy at intervals (only if not game over)
        if not self.game_over and (current_time - self.enemy_timer >= ENEMY_SPAWN_INTERVAL):
            self.enemy_timer = current_time
            self.enemies.append(Enemy())

        # Update enemies and check collisions
        enemies_to_remove = []
        for enemy in self.enemies:
            enemy.update(self.dt)
            if enemy.check_collision(ball_pos):
                enemies_to_remove.append(enemy)
                self.score += 1
                self.explosions.append(Explosion(enemy.pos, current_time))
            if enemy.check_anchor_collision(self.anchor):
                self.game_over = True

        for enemy in enemies_to_remove:
            if enemy in self.enemies:
                self.enemies.remove(enemy)

        # Clean up off-screen enemies
        self.enemies = [e for e in self.enemies if -50 < e.pos.x < WIDTH + 50 and -50 < e.pos.y < HEIGHT + 50]

        # Clean up explosions that have expired
        self.explosions = [exp for exp in self.explosions if exp.is_active(current_time)]

        # Construct a simple observation:
        # For enemy observation, use the nearest enemy relative to the ball.
        if self.enemies:
            distances = [(e.pos - ball_pos).length() for e in self.enemies]
            nearest_enemy = min(self.enemies, key=lambda e: (e.pos - ball_pos).length())
            enemy_rel = nearest_enemy.pos - ball_pos
        else:
            enemy_rel = pygame.math.Vector2(0, 0)

        obs = np.array([self.anchor.x, self.anchor.y,
                        ball_pos.x, ball_pos.y,
                        enemy_rel.x, enemy_rel.y,
                        self.score], dtype=np.float32)

        # Reward: For instance, +1 per enemy hit; game over gives a penalty.
        reward = 1.0 if enemies_to_remove else 0.0
        if self.game_over:
            reward = -10.0

        done = self.game_over

        info = {}

        # If rendering is enabled, draw the current frame.
        if self.render_mode:
            self._render_frame()

        return obs, reward, done, info

    def reset(self):
        self.anchor = ANCHOR_CENTER.copy()
        self.rope = Rope(self.anchor)
        self.enemies = []
        self.explosions = []
        self.score = 0
        self.game_over = False
        self.enemy_timer = pygame.time.get_ticks()
        # Return initial observation with no enemy present.
        ball_pos = self.rope.get_ball_pos()
        obs = np.array([self.anchor.x, self.anchor.y,
                        ball_pos.x, ball_pos.y,
                        0.0, 0.0,
                        self.score], dtype=np.float32)
        return obs

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
