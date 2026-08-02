#include <Arduino.h>

// --- PIN DEFINITIONS ---
const uint8_t PEND_ENC_A = 2; 
const uint8_t PEND_ENC_B = 4;
const uint8_t CART_ENC_A = 3; 
const uint8_t CART_ENC_B = 5;
const uint8_t MOTOR_PWM_L = 9;
const uint8_t MOTOR_PWM_R = 10;
const uint8_t MOTOR_ENA1  = 11;
const uint8_t MOTOR_ENA2  = 12;

// --- GLOBAL VARIABLES ---
volatile long pendPos = 0;
volatile long cartPos = 0;
long targetCartPos = 0;

// --- PENDULUM FILTER VARIABLES ---
float filteredPendPos = 0; 
float filteredPendVel = 0; 
const float kPosPend = 0.7;  // High trust for position to reduce lag
const float kVelPend = 0.1;  // Smooth velocity for the model

// --- CART FILTER VARIABLES ---
float filteredCartPos = 0;
float filteredCartVel = 0;
const float kPosCart = 0.9;  // High trust so the cart stays snappy
const float kVelCart = 0.005;  // Low trust to stop D-term "buzzing"

unsigned long lastLoopTime = 0;
const uint8_t LOOP_INTERVAL = 1; // ~60Hz to match training

// --- TUNING PARAMETERS ---
float Kp = 0.1;         
float Kd = 0.004;        
int minPWM = 0;
int maxPWM = 240;

// --- PWM SLEW LIMITER (caps delta-PWM per loop to reduce jerk) ---
static float constrainedOutput = 0.0;
const float MAX_PWM_CHANGE_PER_LOOP = 3.0; // tune: higher = snappier, lower = smoother

// --- INTERRUPTS & HELPERS ---
void pendISR() { digitalRead(PEND_ENC_A) != digitalRead(PEND_ENC_B) ? pendPos++ : pendPos--; }
void cartISR() { digitalRead(CART_ENC_A) != digitalRead(CART_ENC_B) ? cartPos++ : cartPos--; }

long safeRead(volatile long &sharedVar) {
  noInterrupts();
  long value = sharedVar;
  interrupts();
  return value;
}

void driveMotor(int speed) {
  int absSpeed = abs(speed);
  if (absSpeed > 0 && absSpeed < minPWM) absSpeed = minPWM;
  int pwmValue = constrain(absSpeed, 0, maxPWM);
  
  // Use the raw cartPos for safety limits
  long rawCart = safeRead(cartPos);
  if (rawCart <= -3950 && speed < 0) pwmValue = 0;
  if (rawCart >= 3950 && speed > 0) pwmValue = 0;

  if (speed > 0) {
    analogWrite(MOTOR_PWM_L, 0);
    analogWrite(MOTOR_PWM_R, pwmValue);
  } else if (speed < 0) {
    analogWrite(MOTOR_PWM_R, 0);
    analogWrite(MOTOR_PWM_L, pwmValue);
  } else {
    analogWrite(MOTOR_PWM_L, 0);
    analogWrite(MOTOR_PWM_R, 0);
  }
}

void setup() {
  Serial.begin(115200); 
  pinMode(PEND_ENC_A, INPUT_PULLUP);
  pinMode(PEND_ENC_B, INPUT_PULLUP);
  pinMode(CART_ENC_A, INPUT_PULLUP);
  pinMode(CART_ENC_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PEND_ENC_A), pendISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(CART_ENC_A), cartISR, CHANGE);
  pinMode(MOTOR_PWM_L, OUTPUT); pinMode(MOTOR_PWM_R, OUTPUT);
  pinMode(MOTOR_ENA1, OUTPUT); pinMode(MOTOR_ENA2, OUTPUT);
  digitalWrite(MOTOR_ENA1, HIGH); digitalWrite(MOTOR_ENA2, HIGH);
}

void loop() {
  if (Serial.available() >= 2) {
    int16_t receivedVal;
    Serial.readBytes((char*)&receivedVal, 2);
    targetCartPos = receivedVal;
  }

  unsigned long currentTime = millis();
  if (currentTime - lastLoopTime >= LOOP_INTERVAL) {
    float dt = (float)LOOP_INTERVAL / 1000.0;
    
    // Read raw sensors
    long rawCart = safeRead(cartPos);
    long rawPend = safeRead(pendPos);
    
    // --- PENDULUM FILTER ---
    float predPendPos = filteredPendPos + (filteredPendVel * dt);
    float pendErr = (float)rawPend - predPendPos;
    filteredPendPos = predPendPos + (kPosPend * pendErr);
    filteredPendVel += (kVelPend / dt) * pendErr;

    // --- CART FILTER ---
    float predCartPos = filteredCartPos + (filteredCartVel * dt);
    float cartErr = (float)rawCart - predCartPos;
    filteredCartPos = predCartPos + (kPosCart * cartErr);
    filteredCartVel += (kVelCart / dt) * cartErr;

    // --- PD CONTROL ---
    // Use filtered values for much smoother motor response
    float posError = (float)targetCartPos - filteredCartPos;
    float output = (Kp * posError) - (Kd * filteredCartVel);

    // --- PWM SLEW LIMITER ---
    // Caps how much the driven PWM can change in a single loop, to reduce jerk.
    float outputDelta = output - constrainedOutput;
    outputDelta = constrain(outputDelta, -MAX_PWM_CHANGE_PER_LOOP, MAX_PWM_CHANGE_PER_LOOP);
    constrainedOutput += outputDelta;

    driveMotor((int)constrainedOutput);

    // --- TELEMETRY ---
    // Send filtered values so Python sees exactly what the control loop sees
    int16_t sendPend = (int16_t)filteredPendPos; 
    int16_t sendPendVel = (int16_t)filteredPendVel;
    int16_t sendCart = (int16_t)filteredCartPos; 

    Serial.write('H'); Serial.write('B');
    Serial.write((uint8_t*)&sendPend, 2);
    Serial.write((uint8_t*)&sendPendVel, 2);
    Serial.write((uint8_t*)&sendCart, 2);

    lastLoopTime = currentTime;
  }
}
