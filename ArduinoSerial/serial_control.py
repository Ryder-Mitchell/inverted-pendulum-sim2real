import serial
import time
import numpy as np
import struct
from stable_baselines3 import SAC

# --- CONFIGURATION ---
SERIAL_PORT = 'COM4' 
BAUD_RATE = 115200
MODEL_PATH = "sac_cartpole_model.zip"

class PhysicalCartPole:
    def __init__(self):
        try:
            # We use a 0 timeout because we manage our own timing via get_obs
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0)
            print(f"Connected to Arduino on {SERIAL_PORT}")
        except Exception as e:
            print(f"Error: Could not open serial port. {e}")
            exit()
            
        # Constants
        self.steps_per_meter = 40000.0  
        self.enc_to_rad = (2 * np.pi) / 1200.0
        
        # State Tracking
        self.last_pos = 0.0
        self.last_time = time.time()
        self.last_action = 0.0  
        
        # Wait for Arduino bootloader to finish
        print("Waiting for Arduino...")
        time.sleep(2)
        self.ser.reset_input_buffer()

    def get_obs(self):
        """
        Reads binary data: [H][B][int16 pend][int16 pend_vel][int16 cart]
        Total: 8 bytes
        """
        EXPECTED_SIZE = 8 
            
        # 1. Buffer Management: Always grab the LATEST packet
        # If the Arduino is faster than Python, the buffer fills with old data.
        # This skips old packets to reduce lag.
        if self.ser.in_waiting > EXPECTED_SIZE * 2:
            self.ser.read(self.ser.in_waiting - EXPECTED_SIZE)

        if self.ser.in_waiting < EXPECTED_SIZE:
            return None

        try:
            # 2. Sync with Header
            if self.ser.read(1) != b'H': return None
            if self.ser.read(1) != b'B': return None

            # 3. Read 6 bytes (3 signed shorts)
            raw_payload = self.ser.read(6)
            # '<hhh' = Little-endian, 3x signed short (2 bytes each)
            raw_pend, raw_pend_vel, raw_cart = struct.unpack('<hhh', raw_payload)

            # 4. Conversion to SI Units
            # Note: 600 is your encoder offset for 'straight up'
            theta = (raw_pend - 600) * self.enc_to_rad
            theta_dot = raw_pend_vel * self.enc_to_rad # High-res from Arduino
            pos = raw_cart / self.steps_per_meter
            
            # Simple cart velocity (Arduino could send this too, but this is fine)
            now = time.time()
            dt = now - self.last_time
            if dt <= 0: dt = 0.001
            pos_dot = (pos - self.last_pos) / dt
            
            self.last_pos = pos
            self.last_time = now

            # 5. Observation for SAC (Matches typical Gym CartPole environment)
            return np.array([
                pos,                      # x
                pos_dot,                  # x_dot
                np.sin(theta),            # sin(theta)
                np.cos(theta),            # cos(theta)
                theta_dot,                # theta_dot
                self.last_action          # previous action for smoothness
            ], dtype=np.float32)
        
        except Exception as e:
            return None
    
    def send_action(self, target_meters):
        """Sends target cart position to Arduino PD loop"""
        self.last_action = float(target_meters)
        
        # Convert meters to steps
        target_steps = int(target_meters * self.steps_per_meter)
        
        # Hardware Safety Guard
        target_steps = max(-3850, min(3850, target_steps))
        
        # Pack as 16-bit signed integer
        packet = struct.pack('<h', target_steps) 
        self.ser.write(packet)

def run_live_control():
    robot = PhysicalCartPole()
    
    print(f"Loading SAC model: {MODEL_PATH}")
    try:
        model = SAC.load(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("--- LIVE CONTROL ACTIVE ---")
    
    try:
        while True:
            # Loop runs as fast as Python allows, but get_obs() 
            # only returns data when a full packet arrives.
            obs = robot.get_obs()
            
            if obs is not None:
                # 1. Model Prediction
                action, _ = model.predict(obs, deterministic=True)
                target_m = action[0]
                
                # 2. Update Hardware
                robot.send_action(target_m)

                # 3. Telemetry Printout
                angle_deg = np.degrees(np.arctan2(obs[2], obs[3]))
                print(f"\rTarget: {target_m*100: >6.2f}cm | Pos: {obs[0]*100: >6.2f}cm | Angle: {angle_deg: >6.1f}°", end="")
            
            # Tiny sleep to prevent 100% CPU usage
            time.sleep(0.001)
                        
    except KeyboardInterrupt:
        print("\nStopping... Centering Cart.")
        robot.send_action(0)
    finally:
        robot.ser.close()

if __name__ == "__main__":
    run_live_control()