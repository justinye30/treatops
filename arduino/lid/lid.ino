/*
  TREATOPS: the candy box's lid. An Arduino Uno on the laptop's USB, with the SG90 servo's signal (orange
  wire) on pin 9.

  The laptop (laptop/lid.py) sends one line per command over serial at 115200:
    LID CLOSE   swing the lid down over the candy
    LID OPEN    back up
  and the board answers with the lid's new state ("CLOSED" / "OPEN"). It starts open.
*/
#include <Servo.h>

Servo lidServo;
const int servoPin = 9;    // orange wire
const int openPos = 7;     // 10 degrees closer to the horizon than vertical (0)
const int closedPos = 90;  // swings 83 degrees down to close

void setup() {
  Serial.begin(115200);
  lidServo.attach(servoPin);
  lidServo.write(openPos);
  Serial.println("TREATOPS LID READY");
}

void loop() {
  if (!Serial.available()) return;
  String command = Serial.readStringUntil('\n');
  command.trim();
  if (command == "LID CLOSE") {
    lidServo.write(closedPos);
    Serial.println("CLOSED");
  } else if (command == "LID OPEN") {
    lidServo.write(openPos);
    Serial.println("OPEN");
  }
}
