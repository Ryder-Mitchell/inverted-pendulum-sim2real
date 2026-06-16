# inverted-pendulum-sim2real

Trains a reinforcement learning policy in simulation and runs it on a real cart-pole system built from scratch.

https://www.youtube.com/shorts/UFT4sX_hefM

---

## What it does

A SAC (Soft Actor-Critic) policy is trained in PyBullet to swing up and balance a pendulum on a moving cart. Once trained, the policy runs live on physical hardware — a hand-built cart, Teensy 4.1, and DC motor.

The three pieces:
- `sim.py` — PyBullet gym environment + SAC training (~5-10 min on CPU)
- `ros2.py` — Python node that runs the policy and sends target positions
- `firmware.cpp` — C++ running on the Teensy at 1000Hz, reads encoders, drives the motor

---

## Hardware

- Hand-fabricated cart and rail
- Teensy 4.1
- IBT-2 motor driver
- Quadrature encoders (cart + pendulum)
- Host PC running ROS2 on Ubuntu

---

## A few decisions worth noting

**Switched from force to position control**
The original action space sent force commands directly to the motor. It worked in sim but didn't transfer — PWM and actual force have a nonlinear relationship at low speeds that the simulation didn't capture. Switching the policy output to a target cart position, tracked by a local PD loop on the Teensy, fixed the transfer.

**PD controller first**
Before touching RL, a classical PD controller was tuned to balance the pendulum from near-upright. This validated the hardware and sensor pipeline before adding complexity.

**Alpha filter on the Teensy**
Raw encoder readings are noisy at 1000Hz. A predictor-corrector filter smooths position and velocity estimates for both the cart and pendulum without adding much lag.

---

## Data flow

```
PyBullet (training)
    ↓
ROS2 inference node → /motor/target_steps
    ↓
Teensy 4.1 (1000Hz PD loop) → IBT-2 → Motor
    ↑
Encoders → /encoder/angle_rad
         → /encoder/vel_rad_s
         → /encoder2/linear_ticks
```

---

## Dependencies

**Python:** stable-baselines3, pybullet, gymnasium, numpy, rclpy

**Firmware:** micro_ros_arduino, QuadEncoder

**Platform:** ROS2 (Ubuntu), Teensy 4.1 via micro-ROS USB

---

## Training

```bash
python sim.py
```

Uses 8 parallel workers — adjust to match your physical core count.

---

## Running on hardware

1. Flash `firmware.cpp` to the Teensy
2. Start the micro-ROS agent
3. Run the inference node:

*ROS2 environment must be sourced first

```bash
python ros2.py
```
