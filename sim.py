import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

# --- ENVIRONMENT CLASS ---
class CartPoleSwingUpEnv(gym.Env):
    def __init__(self, render=False, frame_skip=4):
        super(CartPoleSwingUpEnv, self).__init__()
        self.render_mode = render
        self.frame_skip = frame_skip

        self.physicsClient = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Action space: Desired target position of the cart [meters]
        self.action_space = spaces.Box(low=-0.50, high=0.50, shape=(1,), dtype=np.float32)
        
        self.obs_dim = 6
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.last_action = np.zeros(1, dtype=np.float32) 

        self.cartpole_filename = "example-pendulum.urdf"
        self.base_z = 1.5

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05), np.random.uniform(-9.71, -9.91))
        p.loadURDF("plane.urdf")
        
        p.setTimeStep(1/240, self.physicsClient)       
        self.cartpoleId = p.loadURDF(self.cartpole_filename, basePosition=[0, 0, self.base_z])
        
        self.prismatic_index = 0
        self.pole_index = 1
        for i in range(p.getNumJoints(self.cartpoleId)):
            info = p.getJointInfo(self.cartpoleId, i)
            if info[2] == p.JOINT_PRISMATIC: self.prismatic_index = i
            if info[2] == p.JOINT_REVOLUTE: self.pole_index = i

        p.setJointMotorControl2(self.cartpoleId, self.pole_index, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpoleId, self.prismatic_index, p.POSITION_CONTROL, targetPosition=0)

        start_angle = np.random.uniform(-np.pi, np.pi)
        p.resetJointState(self.cartpoleId, self.pole_index, start_angle)
        
        self.last_action = np.zeros(1, dtype=np.float32)
        return self._get_obs(), {}

    def _get_obs(self):
        cart_state = p.getJointState(self.cartpoleId, self.prismatic_index)
        pole_state = p.getJointState(self.cartpoleId, self.pole_index)
        theta = pole_state[0]
        physical_obs = [
            cart_state[0], cart_state[1],
            np.sin(theta), np.cos(theta),
            pole_state[1],
            self.last_action[0] 
        ]
        return np.array(physical_obs, dtype=np.float32)

    def step(self, action):
        target_pos = float(action[0])
        total_reward = 0.0
        terminated = False

        for _ in range(self.frame_skip):
            # Switched to POSITION_CONTROL with a standard 10 N force limit
            p.setJointMotorControl2(
                self.cartpoleId, 
                self.prismatic_index, 
                p.POSITION_CONTROL, 
                targetPosition=target_pos,
                force=7
            )
            p.stepSimulation()

            # State extraction
            cart_pos, cart_vel = p.getJointState(self.cartpoleId, self.prismatic_index)[:2]
            theta, pole_vel = p.getJointState(self.cartpoleId, self.pole_index)[:2]
            
            # 1. Primary Goal: cos(theta)
            # Ranges from -1 (bottom) to 1 (top). shifted to 0-1 and cubed for a steeper peak.
            upright_reward = 2 * ((np.cos(theta) + 1.0) / 2.0) ** 3
            step_reward = upright_reward
            step_reward += 2*((np.cos(theta) + 1.0) / 2.0)
            
            # 2. Stability: Penalize movement only when mostly upright
            if upright_reward > 1.8:
                step_reward -= 0.6 * abs(pole_vel)

            

            # 3. Position & Termination Bounds
            dist = abs(cart_pos)
            if dist > 0.13:
                step_reward -= 5.0 
            
            if dist > 0.15:
                total_reward = -100.0
                terminated = True
                break
            
            step_reward += 100 * (0.13 - dist)
            total_reward += step_reward

        # 4. Hardware Safety: Smoothing & Direction Flip mapped to Position Commands
        # Target position adjustment penalty (prevents high-frequency target jitter)
        total_reward -= 0.05 * abs(target_pos - self.last_action[0])
        
        # Zero-crossing penalty (prevents crossing the center line aggressively back and forth)
        #if target_pos * self.last_action[0] < 0:
        #    total_reward -= 0.5 

        self.last_action = np.array(action, dtype=np.float32)
        return self._get_obs(), total_reward, terminated, False, {}

# --- HELPERS ---
def make_env(rank, seed=0, log_dir=None):
    def _init():
        env = CartPoleSwingUpEnv(render=False, frame_skip=4)
        env = TimeLimit(env, max_episode_steps=1000)
        if log_dir is not None:
            env = Monitor(env, os.path.join(log_dir, str(rank)))
        return env
    set_random_seed(seed)
    return _init

# --- EXECUTION ---
if __name__ == "__main__":
    log_dir = "./sac_logs_clean/"
    os.makedirs(log_dir, exist_ok=True)

    # 1. TRAINING PHASE
    train_env = SubprocVecEnv([make_env(i, log_dir=log_dir) for i in range(8)])
    
    model = SAC(
        "MlpPolicy", 
        train_env, 
        verbose=1, 
        learning_rate=3e-4,     
        batch_size=256, 
        train_freq=1, 
        gradient_steps=1,
        ent_coef="auto_0.1"     
    )
    
    print("Training policy using Position Control...")
    model.learn(total_timesteps=300000)
    model.save("sac_cartpole_optimized")
    train_env.close()

    # 2. VISUALIZATION PHASE
    print("Testing learned policy...")
    test_env = CartPoleSwingUpEnv(render=True, frame_skip=2)
    test_env = TimeLimit(test_env, max_episode_steps=1000)
    obs, _ = test_env.reset()
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = test_env.step(action)
            time.sleep(1/60) 
            if done or truncated: obs, _ = test_env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        test_env.close()
