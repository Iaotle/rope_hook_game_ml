import os
import time
import numpy as np
import torch
from datetime import datetime
import signal
import sys
import pygame
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym

from RopeDefenseEnv import RopeEnv

# Create directories for logs and models
log_dir = "logs"
model_dir = "models/sac"
best_model_dir = os.path.join(model_dir, "best")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(best_model_dir, exist_ok=True)

# Create a unique timestamp for this training run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"sac_rope_defense_{timestamp}"

# Set up environment creation function
def make_env(render=False):
    """Create and return a wrapped environment instance"""
    env = RopeEnv(render_mode=render)
    env = Monitor(env)
    return env

# Create vectorized environment for training (no rendering)
print("Setting up training environment...")
from stable_baselines3.common.env_util import make_vec_env
env = make_vec_env(make_env, n_envs=8, )
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Check for GPU availability
if torch.cuda.is_available():
    device = "cuda"
    gpu_info = torch.cuda.get_device_name(0)
    print(f"Training on GPU: {gpu_info}")
else:
	device = "cpu"
	print("No GPU detected, training on CPU")

# Define important paths for saving/loading models and normalization statistics
best_model_path = os.path.join(best_model_dir, "best_model")
final_model_path = os.path.join(model_dir, f"final_model_{run_name}")
interrupt_model_path = os.path.join(model_dir, f"interrupt_model_{run_name}")
vecnorm_path = os.path.join(model_dir, f"vecnorm_{run_name}.pkl")

from stable_baselines3.common.utils import get_schedule_fn

# Define a cosine annealing learning rate schedule
initial_learning_rate = 3e-4
def cosine_annealing_schedule(progress_remaining):
    return initial_learning_rate * (1 + np.cos(np.pi * progress_remaining)) / 2

# Create a new SAC model if no best checkpoint exists
print("Initializing new SAC model...")
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=get_schedule_fn(cosine_annealing_schedule),  # Use the learning rate schedule
    buffer_size=1_000_000,   # size of the replay buffer
    batch_size=1024,
    tau=0.01,
    # learning_starts=2_000_000, # will actually just be retarded, since the data it collects will be garbage
    gamma=0.99,
    train_freq=1,          # train every step
    gradient_steps=4,
    use_sde=True,
    sde_sample_freq=4,
    target_update_interval=2,
    verbose=1,
    tensorboard_log=log_dir,
    device=device,
    # policy_kwargs={"net_arch": {"pi": [256, 256], "qf": [256, 256]}}
    policy_kwargs = {"net_arch": {"pi": [256, 256, 256], "qf": [256, 256, 256]},
                        "activation_fn": torch.nn.SiLU,
                        "log_std_init": -0.5}
    

)
model.metadata = {
    "run_name": run_name,
    "timestamp": timestamp,
}

# Setup evaluation environment
eval_env = DummyVecEnv([lambda: make_env()])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)
eval_env.obs_rms = env.obs_rms
eval_env.ret_rms = env.ret_rms

# Configure callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=model_dir,
    name_prefix=run_name,
    verbose=1
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=best_model_dir,
    log_path=log_dir,
    eval_freq=10000,
    deterministic=True,
    render=False,
    verbose=1
)

from stable_baselines3.common.callbacks import BaseCallback

class StopOnPlateauCallback(BaseCallback):
    """
    Stop training when the model stops improving for a specified number of evaluations.
    """
    def __init__(
        self, 
        eval_callback, 
        patience=5, 
        min_timesteps=100000, 
        min_evals=5, 
        improvement_threshold=0.01,
        verbose=1
    ):
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.patience = patience
        self.min_timesteps = min_timesteps
        self.min_evals = min_evals
        self.improvement_threshold = improvement_threshold
        self.best_mean_reward = -float('inf')
        self.no_improvement_count = 0
        
    def _on_step(self) -> bool:
        # Skip if we haven't done enough timesteps
        if self.num_timesteps < self.min_timesteps:
            return True
            
        # Only check after eval_callback has done an evaluation
        if not hasattr(self.eval_callback, 'last_mean_reward') or self.eval_callback.last_mean_reward is None:
            return True
            
        # Skip until we've done enough evaluations
        if not hasattr(self.eval_callback, 'evaluations_done') or self.eval_callback.evaluations_done < self.min_evals:
            return True
            
        current_reward = self.eval_callback.last_mean_reward
        
        # Check if there's a significant improvement
        if current_reward > self.best_mean_reward + self.improvement_threshold:
            self.best_mean_reward = current_reward
            self.no_improvement_count = 0
            if self.verbose > 0:
                print(f"New best mean reward: {self.best_mean_reward:.2f}")
        else:
            self.no_improvement_count += 1
            if self.verbose > 0:
                print(f"No significant improvement for {self.no_improvement_count} evaluations. Current: {current_reward:.2f}, Best: {self.best_mean_reward:.2f}")
                
        # Stop training if we've had no improvement for a while
        if self.no_improvement_count >= self.patience:
            if self.verbose > 0:
                print(f"Stopping training due to no improvement for {self.patience} evaluations.")
            return False
            
        return True

# Add the plateau detection callback
plateau_callback = StopOnPlateauCallback(
    eval_callback=eval_callback,
    patience=5,              # Stop after 5 evals without improvement
    min_timesteps=100000,    # Train for at least 100k steps
    min_evals=5,             # Do at least 5 evaluations
    improvement_threshold=0.01, # Reward must improve by at least 0.01
    verbose=1
)

class BestModelVisualizationCallback(BaseCallback):
    """
    Callback to periodically visualize the best model during training.
    """
    def __init__(self, best_model_dir, eval_freq=10000, preview_steps=500, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.best_model_dir = best_model_dir
        self.best_model_path = os.path.join(best_model_dir, "best_model")
        self.preview_steps = preview_steps
        self.last_viz_timestep = 0
        
    def _on_step(self) -> bool:
        # Check if it's time to visualize
        if (self.num_timesteps - self.last_viz_timestep) < self.eval_freq:
            return True
            
        # Update last visualization timestep
        self.last_viz_timestep = self.num_timesteps
        
        # Check if best model exists
        if not os.path.exists(f"{self.best_model_path}.zip"):
            if self.verbose > 0:
                print(f"No best model found yet at {self.best_model_path}.zip")
            return True
            
        if self.verbose > 0:
            print(f"\n{'-'*40}")
            print(f"Visualizing best model at {self.num_timesteps} steps")
            print(f"{'-'*40}")
        
        try:
            # Create rendering environment
            render_env = RopeEnv(render_mode=True, simulation_fps=60)
            # Load the best model
            best_model = SAC.load(self.best_model_path)
            
            # Run the model in the environment
            obs, _ = render_env.reset()
            done = False
            total_reward = 0
            step_count = 0
            
            while not done and step_count < self.preview_steps:
                action, _ = best_model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = render_env.step(action)
                done = done or truncated
                total_reward += reward
                step_count += 1
                
                # Process pygame events to allow window closing
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
            
            if self.verbose > 0:
                print(f"Visualization complete - Score: {info['score']}")
                print(f"{'-'*40}\n")
                
        except Exception as e:
            print(f"Visualization error: {e}")
        finally:
            render_env.close()
            
        return True

# Add the visualization callback
viz_callback = BestModelVisualizationCallback(
    best_model_dir=best_model_dir,
    eval_freq=500000,  # Show visualization every 500k steps
    preview_steps=500,  # Show 500 steps in each visualization
    verbose=1
)

# Training loop
print(f"Starting training with SAC on {device}...")
print(f"Models will be saved to: {model_dir}")
print(f"Best model will be saved to: {best_model_dir}")
print(f"Run with 'tensorboard --logdir={log_dir}' to view training metrics")
print("Press Ctrl+C at any time to interrupt (model will be saved)")

total_timesteps = 40_000_000  # Adjust as needed
start_time = time.time()
try:
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, plateau_callback, viz_callback, checkpoint_callback],  # checkpoint_callback can be added if desired
        tb_log_name=run_name,
        progress_bar=True
    )
    # Save the final model and normalization stats
    model.save(final_model_path)
    env.save(vecnorm_path)
    print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
    print(f"Final model saved to {final_model_path}")
    model_to_load = final_model_path

except KeyboardInterrupt:
    # Handle manual interruption
    print(f"\nTraining interrupted after {time.time() - start_time:.2f} seconds")
    # Update metadata to indicate interruption
    if hasattr(model, "metadata"):
        model.metadata["interrupted"] = True
        model.metadata["interrupt_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        model.metadata = {"interrupted": True, "interrupt_time": datetime.now().strftime("%Y%m%d_%H%M%S")}
    model.save(interrupt_model_path)
    env.save(vecnorm_path)
    print(f"Interrupted model saved to {interrupt_model_path}")
    model_to_load = interrupt_model_path

# Clean up training environments
env.close()
eval_env.close()

# Determine which model to preview
if os.path.exists(f"{best_model_path}.zip"):
    model_to_load = best_model_path
    print("\nLoading best model for preview...")
elif os.path.exists(f"{model_to_load}.zip"):
    print(f"\nLoading last saved model for preview...")
else:
    print("\nNo model available for preview.")
    sys.exit(0)

# Run visualization with the trained model
print("\nStarting visualization with trained model...")
print("Close the window to exit preview")

# Create rendering environment and load model
render_env = RopeEnv(render_mode=True, simulation_fps=60)
# For the preview, we can use the original RopeEnv directly without wrapping
trained_model = SAC.load(model_to_load)

# Run the model in the environment
obs, _ = render_env.reset()
done = False
total_reward = 0
step_count = 0
max_preview_steps = 10000  # Limit preview length

try:
    while not done and step_count < max_preview_steps:
        # When using with the original env (not wrapped), need to adapt the predict call
        action, _ = trained_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = render_env.step(action)
        done = done or truncated
        total_reward += reward
        step_count += 1
        
        # Show periodic updates
        if step_count % 100 == 0:
            print(f"Step: {step_count}, Score: {info['score']}, Total reward: {total_reward:.2f}")
        
        # Process pygame events to allow window closing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
    
    print(f"\nPreview complete")
    print(f"Final score: {info['score']}")
    print(f"Total steps: {step_count}")
    
except Exception as e:
    print(f"Preview error: {e}")
finally:
    render_env.close()

print("\nTraining session complete!")
