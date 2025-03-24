import pygame
import sys
import math

# Simulation parameters
WIDTH, HEIGHT = 800, 600
FPS = 60
GRAVITY = pygame.math.Vector2(0, 980)  # pixels/s^2
DT = 1 / FPS
NUM_SEGMENTS = 20       # Number of rope segments
SEGMENT_LENGTH = 20     # Length of each rope segment in pixels
ITERATIONS = 10         # Constraint iterations for stability
DAMPING = 0.99          # Damping factor for Verlet integration

# Limit for the anchor movement (a circle centered in the middle)
ANCHOR_CENTER = pygame.math.Vector2(WIDTH // 2, HEIGHT // 2)
ANCHOR_RADIUS = 250    # Allowed radius in pixels

def clamp_to_circle(point, center, radius):
    """Clamp the given point to be within a circle of given center and radius."""
    vec = point - center
    if vec.length() > radius:
        vec.scale_to_length(radius)
        return center + vec
    return point

class Rope:
    def __init__(self, anchor_pos, num_segments=NUM_SEGMENTS, segment_length=SEGMENT_LENGTH):
        self.num_segments = num_segments
        self.segment_length = segment_length
        # Create a list of points, each with a current and previous position.
        self.points = []
        for i in range(num_segments + 1):
            pos = pygame.math.Vector2(anchor_pos[0], anchor_pos[1] + i * segment_length)
            self.points.append({
                'pos': pos.copy(),
                'prev_pos': pos.copy()
            })

    def verlet(self):
        # Update all points except the anchor (index 0) using Verlet integration.
        for i in range(1, len(self.points)):
            point = self.points[i]
            current = point['pos']
            prev = point['prev_pos']
            velocity = (current - prev) * DAMPING
            point['prev_pos'] = current.copy()
            point['pos'] = current + velocity + GRAVITY * (DT * DT)

    def apply_constraints(self, anchor_pos):
        # Lock the first point to the clamped anchor position.
        self.points[0]['pos'] = pygame.math.Vector2(anchor_pos)

        # Enforce segment constraints through several iterations.
        for _ in range(ITERATIONS):
            for i in range(len(self.points) - 1):
                p1 = self.points[i]['pos']
                p2 = self.points[i+1]['pos']
                delta = p2 - p1
                dist = delta.length()
                if dist == 0:
                    continue
                diff = (dist - self.segment_length) / dist
                if i == 0:
                    p2 -= delta * diff
                else:
                    p1 += delta * (diff * 0.5)
                    p2 -= delta * (diff * 0.5)
                self.points[i]['pos'] = p1
                self.points[i+1]['pos'] = p2

    def update(self, anchor_pos):
        # Update the physics and then enforce the rope constraints.
        self.verlet()
        self.apply_constraints(anchor_pos)

    def get_points(self):
        # Return a list of positions for drawing.
        return [p['pos'] for p in self.points]

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Flexible Rope with Limited Anchor Movement")
    clock = pygame.time.Clock()

    # Initially, start with the anchor at the center.
    anchor = ANCHOR_CENTER
    rope = Rope(anchor)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get current mouse position and clamp it to the allowed circle.
        mouse_pos = pygame.math.Vector2(pygame.mouse.get_pos())
        clamped_anchor = clamp_to_circle(mouse_pos, ANCHOR_CENTER, ANCHOR_RADIUS)
        
        # Update rope simulation with the clamped anchor.
        rope.update(clamped_anchor)

        # Drawing
        screen.fill((30, 30, 30))  # Dark background

        # Draw allowed anchor movement circle
        pygame.draw.circle(screen, (50, 50, 50), (int(ANCHOR_CENTER.x), int(ANCHOR_CENTER.y)), ANCHOR_RADIUS, 2)
        
        # Draw the rope segments.
        points = rope.get_points()
        for i in range(len(points) - 1):
            pygame.draw.line(screen, (200, 200, 200),
                             (int(points[i].x), int(points[i].y)),
                             (int(points[i+1].x), int(points[i+1].y)), 2)
        # Draw the ball (last point).
        ball_pos = points[-1]
        pygame.draw.circle(screen, (255, 100, 100), (int(ball_pos.x), int(ball_pos.y)), 15)
        
        # Draw the anchor (clamped mouse position).
        pygame.draw.circle(screen, (100, 255, 100), (int(clamped_anchor.x), int(clamped_anchor.y)), 5)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
