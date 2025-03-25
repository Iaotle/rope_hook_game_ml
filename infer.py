import os
import sys
import time
import pygame
from stable_baselines3 import PPO
from RopeDefenseEnv import RopeEnv

# Path to the saved best model
best_model_path = "models/best/best_model.zip"

# Make sure the best model exists before proceeding
if not os.path.exists(best_model_path):
    print(f"Error: Best model not found at {best_model_path}")
    sys.exit(1)

# Load the best model
print(f"Loading best model from {best_model_path}...")
model = PPO.load(best_model_path)

# Create a rendering environment (similar to the one used during training)
render_env = RopeEnv(render_mode=True, simulation_fps=60)

# Run the model for a fixed number of steps or until the user quits
max_steps = 10000  # Adjust as needed, for demo purposes
step_count = 0

print("\nStarting visualization with the trained model...\n")
print("Close the window to exit preview")

# Run inference loop
done = False
obs = render_env.reset()
total_reward = 0

try:
    while not done and step_count < max_steps:
        # Predict the next action using the trained model
        action, _ = model.predict(obs, deterministic=True)
        
        # Step the environment with the action
        obs, reward, done, info = render_env.step(action)
        total_reward += reward
        step_count += 1
        
        # Periodically output progress to the console
        if step_count % 100 == 0:
            print(f"Step: {step_count}, Score: {info['score']}, Total reward: {total_reward:.2f}")
        
        # Process pygame events to allow window closing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

except Exception as e:
    print(f"Error during inference: {e}")
finally:
    render_env.close()

print(f"\nInference complete.")
print(f"Final score: {info['score']}")
print(f"Total steps: {step_count}")
