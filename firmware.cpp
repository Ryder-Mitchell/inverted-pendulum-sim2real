#include <Arduino.h>
#include <micro_ros_arduino.h>
#include <QuadEncoder.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <std_msgs/msg/float64.h>
#include <std_msgs/msg/int16.h>

// --- HARDWARE CONFIGURATION (Swapped Register Modules) ---
QuadEncoder hardware_encoder2(2, 2, 3, 1);   // Pendulum Encoder (Module 2 -> Pins 2 & 3)
QuadEncoder hardware_encoder(1, 0, 1, 1);  // Cart Encoder (Module 1 -> Pins 0 & 1)

const double TICKS_PER_REV = 2400.0;        
const double TICKS_PER_METER = 40000.0;     

// IBT-2 Motor Driver Pins
const int RPWM_PIN = 6;
const int LPWM_PIN = 9;
const int ENABLE_PIN = 7;

// --- TARGET POSITION COMMITTED BY ROS2 NODE ---
volatile int16_t targetCartPos = 0; 

// --- STATE FILTERING VARIABLES (Matching Original Alpha Coefficients) ---
float filteredPendPos = 0.0; 
float filteredPendVel = 0.0; 
float filteredCartPos = 0.0; 
float filteredCartVel = 0.0; 

// --- LOCAL METRIC PD TRACKING ENGINE GAINS (Scaled Up for 12-Bit Resolution) ---
// Since maxPWM is ~17x larger (4095 vs 240), the base multipliers are scaled proportionally
float Kp = 0.4 * TICKS_PER_METER;  
float Kd = -0.05 * TICKS_PER_METER;       
int maxPWM = 7700; // 12-bit with cap
float lastOutput = 0;
// --- TORQUE JERK SLEW FILTER (Scaled Up for 12-Bit Resolution) ---
static float constrainedOutput = 0.0;
// 3.0 units at 8-bit scales to roughly 51.0 units at 12-bit resolution to retain the same torque ramp
const float MAX_PWM_CHANGE_PER_LOOP = 80.0; 

// --- MICRO-ROS DATA STRUCTURE MANIFEST ---
rcl_node_t node;
rclc_support_t support;
rcl_allocator_t allocator;
rclc_executor_t executor;

rcl_publisher_t publisher_angle;
rcl_publisher_t publisher_vel;
rcl_publisher_t publisher_ticks;

std_msgs__msg__Float64 msg_angle;
std_msgs__msg__Float64 msg_vel;
std_msgs__msg__Float64 msg_ticks;

rcl_subscription_t subscriber_target;
std_msgs__msg__Int16 msg_target;

void driveMotor(int speed) {
  int pwmValue = constrain(abs(speed), 0, maxPWM);
 
  // Safety Interlocks utilizing raw register evaluation
  int32_t rawCart = -hardware_encoder2.read(); 
  if (rawCart <= -15000 && speed < 0) pwmValue = 0;
  if (rawCart >= 15000 && speed > 0) pwmValue = 0;

  if (speed > 0) {
    analogWrite(LPWM_PIN, 0);
    analogWrite(RPWM_PIN, pwmValue);
  } else if (speed < 0) {
    analogWrite(RPWM_PIN, 0);
    analogWrite(LPWM_PIN, pwmValue);
  } else {
    analogWrite(RPWM_PIN, 0);
    analogWrite(LPWM_PIN, 0);
  }
}

void target_callback(const void *msgin) {
  const std_msgs__msg__Int16 * incoming_msg = (const std_msgs__msg__Int16 *)msgin;
  targetCartPos = incoming_msg->data;
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  
  pinMode(RPWM_PIN, OUTPUT);
  pinMode(LPWM_PIN, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);
  
  // --- CONFIGURE 20KHZ ULTRASONIC FREQUENCY & 12-BIT PWM RESOLUTION ---
  analogWriteFrequency(RPWM_PIN, 20000);
  analogWriteFrequency(LPWM_PIN, 20000);
  analogWriteResolution(13); 
  
  analogWrite(RPWM_PIN, 0);
  analogWrite(LPWM_PIN, 0);
  digitalWrite(ENABLE_PIN, HIGH);

  hardware_encoder.setInitConfig();
  hardware_encoder.init();
  hardware_encoder2.setInitConfig();
  hardware_encoder2.init();

  hardware_encoder.write(0);
  hardware_encoder2.write(0);

  set_microros_transports();
  delay(2000); 

  allocator = rcl_get_default_allocator();
  rclc_support_init(&support, 0, NULL, &allocator);
  rclc_node_init_default(&node, "teensy_cartpole_node", "", &support);

  rclc_publisher_init_default(&publisher_angle, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float64), "/encoder/angle_rad");
  rclc_publisher_init_default(&publisher_vel, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float64), "/encoder/vel_rad_s");
  rclc_publisher_init_default(&publisher_ticks, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float64), "/encoder2/linear_ticks");

  rclc_subscription_init_best_effort(&subscriber_target, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int16), "/motor/target_steps");

  rclc_executor_init(&executor, &support.context, 1, &allocator);
  rclc_executor_add_subscription(&executor, &subscriber_target, &msg_target, &target_callback, ON_NEW_DATA);

  digitalWrite(LED_BUILTIN, HIGH);
}

void loop() {
  static unsigned long last_time = 0;
  static int telemetry_counter = 0;
  unsigned long current_time = millis();

  rclc_executor_spin_some(&executor, 0);

  // Core Math & Direct Motor Control Loop running strictly at 1000Hz (1ms)
  if (current_time - last_time >= 1) { 
    float dt = (float)(current_time - last_time) / 1000.0;
    last_time = current_time;

    int32_t rawPend = -hardware_encoder.read();
    int32_t rawCart = hardware_encoder2.read();

    float rawCartMeters = (float)rawCart / TICKS_PER_METER;
    float rawPendRad = ((float)rawPend / TICKS_PER_REV) * 2.0 * M_PI;

    // --- CART ALPHA STATE OBSERVATION ---
    float predCartPos = filteredCartPos + (filteredCartVel * dt);
    float cartErr = rawCartMeters - predCartPos;
    filteredCartPos = predCartPos + (0.9f * cartErr);
    filteredCartVel += (0.005f / dt) * cartErr;

    // --- PENDULUM ALPHA STATE OBSERVATION ---
    float predPendPos = filteredPendPos + (filteredPendVel * dt);
    float pendErr = rawPendRad - predPendPos;
    while (pendErr > M_PI)  pendErr -= 2.0 * M_PI;
    while (pendErr <= -M_PI) pendErr += 2.0 * M_PI;
    filteredPendPos = predPendPos + (0.7f * pendErr);
    filteredPendVel += (0.1f / dt) * pendErr;

    // --- DIRECT LOCAL TRACKING METRIC ERROR ---
    float rawTargetMeters = (float)targetCartPos / TICKS_PER_METER;
    float posError = rawTargetMeters - filteredCartPos;
    float rawOutput = (Kp * posError) - (Kd * filteredCartVel);

    // --- TORQUE RATE FILTER ---
    float outputDelta = rawOutput - constrainedOutput;
    outputDelta = constrain(outputDelta, -MAX_PWM_CHANGE_PER_LOOP, MAX_PWM_CHANGE_PER_LOOP);
    constrainedOutput += outputDelta;
   
    driveMotor((int)constrainedOutput * 0.3 + lastOutput * 0.7);
    lastOutput = (int)constrainedOutput;
    // --- TELEMETRY TRANSPORT THROTTLE (Currently 0 for maximum frequency)
    telemetry_counter++;
    if (telemetry_counter >= 0) {
      telemetry_counter = 0;

      double bounded_angle = fmod((double)filteredPendPos, 2.0 * M_PI);
      if (bounded_angle < 0) {
        bounded_angle += 2.0 * M_PI;
      }

      msg_angle.data = bounded_angle;       
      msg_vel.data   = (double)filteredPendVel; 
      msg_ticks.data = (double)filteredCartPos; 

      (void) rcl_publish(&publisher_angle, &msg_angle, NULL);
      (void) rcl_publish(&publisher_vel, &msg_vel, NULL);
      (void) rcl_publish(&publisher_ticks, &msg_ticks, NULL);
    }
  }
}
