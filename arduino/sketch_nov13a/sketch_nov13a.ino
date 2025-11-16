/*
  This code defines and initializes two pins for a centrifugal pump and sets 
  the speed of one of the pump pins to HIGH and the other to LOW, causing the 
  pump to spin in one direction. After 5 seconds, the pump turns off.

  Board: Arduino Uno R4 (or R3)
  Component: Motor and L9110 motor control board
*/

// Define the pump pins
const int motorB_1A = 9;
const int motorB_2A = 10;

// speed in the range [-1, 1]
void setMotorSpeed(float speed){
  int val1, val2;
  if (speed > 0) {
    val1 = speed * 255;
    val2 = 0;
  }
  else {
    val1 = 0;
    val2 = -speed*255;
  }

  analogWrite(motorB_1A, val1);
  analogWrite(motorB_2A, val2);
}

void setup() {
  pinMode(motorB_1A, OUTPUT);  // set pump pin 1 as output
  pinMode(motorB_2A, OUTPUT);  // set pump pin 2 as output

  analogWrite(motorB_1A, 128);
  analogWrite(motorB_2A, 0);
}

float t = 0;
void loop() {
  delay(1000/60);
  setMotorSpeed(sin(t/2));

  // setMotorSpeed(-1);
  t += 0.16;
}