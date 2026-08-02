# inverted-pendulum-sim2real

Trains a reinforcement learning policy in simulation and runs it on a real cart-pole system built from scratch.

https://www.youtube.com/shorts/UFT4sX_hefM

---

## What it does

A SAC (Soft Actor-Critic) policy is trained in PyBullet to swing up and balance a pendulum on a moving cart. Once trained, the policy runs live on physical hardware — a hand-built cart, an Arduino-compatible board, and a DC motor driven through an IBT-2 driver.

The three pieces:
- `sim.py` — PyBullet gym environment + SAC training (~5-10 min on CPU)
- `serial_control.py` — Python host script: runs the trained policy and talks to the firmware over USB serial
- `firmware.ino` — runs on the microcontroller: reads the encoders, filters the signal, closes a local PD position loop, and drives the motor

---

## Hardware

- Hand-fabricated cart and rail
- Arduino-compatible microcontroller (Uno/Nano-class — uses `attachInterrupt`, `analogWrite` on pins 9-12)
- IBT-2 motor driver, motor run at 7V
- Quadrature encoders (cart + pendulum)
- Host PC connected via USB serial

---

## A few decisions worth noting

**Switched from force to position control**
The original action space sent force commands directly to the motor. It worked in sim but didn't transfer — PWM and actual force have a nonlinear relationship at low speeds that the simulation didn't capture. Switching the policy output to a target cart position, tracked by a local PD loop on the microcontroller, fixed the transfer.

**PD controller first**
Before touching RL, a classical PD controller was tuned to balance the pendulum from near-upright. This validated the hardware and sensor pipeline before adding complexity.

**Predictor-corrector filter on the microcontroller**
Raw encoder readings are noisy at high loop rates. A predictor-corrector filter (`firmware.ino`: `filteredPendPos`/`filteredPendVel`, `filteredCartPos`/`filteredCartVel`) smooths position and velocity estimates for both the cart and pendulum without adding much lag. The filtered values — not the raw encoder counts — are both what the PD loop controls against and what gets sent back to Python, so the model always sees exactly what the controller sees.

---

## Data flow

```
PyBullet (training)
        |
        v
serial_control.py (host) --- target position (int16, USB serial) --->  firmware.ino
        ^                                                                   |
        |                                                                   v
        +-------------- filtered pendulum + cart state (USB serial) -------+
                                                                    (drives motor via
                                                                     PD loop, encoders
                                                                     feed the filter)
```

Wire format (all little-endian):
- **Host → device:** 2 bytes, signed 16-bit target cart position (encoder ticks)
- **Device → host:** `'H'`, `'B'`, then 3x signed 16-bit values — filtered pendulum position, filtered pendulum velocity, filtered cart position

---

## Dependencies

**Python:** see `requirements.txt` (stable-baselines3, pybullet, gymnasium, numpy, pyserial)

**Firmware:** Arduino core only — no external libraries required

---

## Calibration

Before running on your own hardware, check these constants in `serial_control.py` and `firmware.ino` against your build:

- `steps_per_meter` / cart tick-to-meter conversion — depends on your rail's encoder resolution
- `enc_to_rad` — depends on your pendulum encoder's counts-per-revolution (currently assumes 1200, i.e. 2x-decoded from a 600-tick encoder)
- The `- 600` pendulum offset in `get_obs()` — this zeroes "straight up" for this specific build; recalibrate for yours
- `CART_LIMIT_LEFT` / `CART_LIMIT_RIGHT` in firmware — your rail's physical travel limits, in encoder ticks
- `Kp` / `Kd` in firmware — the on-device PD gains for tracking a target position; tune these on the bench before running the full policy
- `maxPWM` in firmware — this project runs the motor at 7V; if you run at a different supply voltage, the torque delivered at a given PWM duty cycle will differ, so re-tune `maxPWM` and the PD gains accordingly

---

## Training

```bash
pip install -r requirements.txt
python sim.py
```

Uses 8 parallel workers — adjust to match your physical core count. Saves the trained policy as `sac_cartpole_model.zip`.

A pretrained example model is included in this repo under that same filename, so you can skip straight to hardware testing below without training first. It was trained on this project's specific reward function and hardware constants (see Calibration) — swap in your own trained model if your build differs.

---

## Running on hardware

1. Flash `firmware.ino` to your microcontroller
2. Set `SERIAL_PORT` in `serial_control.py` to match your device
3. Run:

```bash
python serial_control.py
```

Start with the cart free to move at low speed before trusting it with a full swing-up — the PD gains above are tuned for this specific build and will need adjustment for yours.

---

## Roadmap

A ROS2 + micro-ROS version (Teensy 4.1, `rclc` executor, topic-based comms instead of raw serial) is in progress and will be published as a separate branch/version when it's ready for others to run.

