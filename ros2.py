#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Int16
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import time
from stable_baselines3 import SAC

MODEL_PATH = "sac_cartpole_optimized.zip"

class Ros2CartpoleRlNode(Node):
    def __init__(self):
        super().__init__('ros2_cartpole_rl_node')
        
        self.steps_per_meter = 40000.0  
        
        # Scaling component maps standard 3850 action boundaries cleanly to physical hardware limit
        self.action_scale_factor = 12000.0 / 3850.0 
        
        # Inward telemetry registers
        self.current_angle_rad = 0.0
        self.current_vel_rad_s = 0.0
        self.current_cart_meters = 0.0
        self.last_pos = 0.0
        self.last_time = time.time()
        self.last_action = 0.0

        self.get_logger().info(f"Accessing SAC Weight Matrix from: {MODEL_PATH}")
        try:
            self.model = SAC.load(MODEL_PATH, device="cpu")
            self.get_logger().info("SAC Weight File Parsed Successfully.")
        except Exception as e:
            self.get_logger().error(f"Critical System Failure: Execution weights unreadable: {e}")
            raise e

        # Real-time network execution policy mapping
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Communication links
        self.sub_angle = self.create_subscription(Float64, '/encoder/angle_rad', self.angle_callback, qos)
        self.sub_vel   = self.create_subscription(Float64, '/encoder/vel_rad_s', self.vel_callback, qos)
        self.sub_ticks = self.create_subscription(Float64, '/encoder2/linear_ticks', self.ticks_control_trigger_callback, qos)
        
        self.pub_target = self.create_publisher(Int16, '/motor/target_steps', qos)

        self.get_logger().info("--- INFERENCE NODE ACTIVE (ZERO-EMA) ---")

    def angle_callback(self, msg):
        self.current_angle_rad = msg.data

    def vel_callback(self, msg):
        self.current_vel_rad_s = msg.data

    def ticks_control_trigger_callback(self, msg):
        self.current_cart_meters = msg.data

        now = time.time()
        dt = now - self.last_time
        if dt <= 0: dt = 0.001
            
        pos = self.current_cart_meters/1.6
        pos_dot = (pos - self.last_pos) / dt
        
        self.last_pos = pos
        self.last_time = now

        # Convert continuous angle down to Gym environment space (-PI to +PI tracking)
        calibrated_angle = self.current_angle_rad - np.pi
        theta_dot = self.current_vel_rad_s

        # Package state space exactly to environment definitions
        obs = np.array([
            pos,                                 
            pos_dot,                             
            np.sin(calibrated_angle),            
            np.cos(calibrated_angle),            
            theta_dot,                           
            self.last_action                     
        ], dtype=np.float32)

        # Predictive inference - Output is used immediately
        action, _ = self.model.predict(obs, deterministic=True)
        target_meters = action[0]

        self.last_action = float(target_meters)

        # Scale raw target directly into step parameters without EMA smoothing lag
        target_steps = int(target_meters * self.steps_per_meter * self.action_scale_factor)
        target_steps = max(-15000, min(15000, target_steps))

        action_msg = Int16()
        action_msg.data = target_steps
        self.pub_target.publish(action_msg)

        # Display terminal tracking diagnostic metrics
        print(f"\rTarget: {target_steps: >5} steps | Pos: {pos: >6.3f}m | Angle: {np.degrees(calibrated_angle): >6.1f}°", end="", flush=True)

def main(args=None):
    rclpy.init(args=args)
    node = Ros2CartpoleRlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("\nInterrupt caught. Issuing structural recovery center command...")
        emergency_msg = Int16()
        emergency_msg.data = 0
        node.pub_target.publish(emergency_msg)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()