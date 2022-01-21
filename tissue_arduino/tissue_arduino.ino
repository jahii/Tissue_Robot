#include <Adafruit_NeoPixel.h>
#include <string.h>
#include <Servo.h>
Servo motor_t;
Servo motor_c;
int tissue_pin = 3;
int cutting_pin = 5;
int dt_pin_tri = 8;
int dt_pin_ech = 9;
int ts_pin_tri = 10;
int ts_pin_ech = 11;
int n_led = 12;
int neo_pin = 6;
int st = 86;

Adafruit_NeoPixel strip = Adafruit_NeoPixel(n_led, neo_pin, NEO_GRB + NEO_KHZ800);

void setup() {
  Serial.begin(9600);
  pinMode(13, OUTPUT);
  pinMode(dt_pin_tri, OUTPUT);
  pinMode(dt_pin_ech, INPUT);

  motor_t.write(st);
  motor_c.write(st);
  pinMode(ts_pin_tri, OUTPUT);
  pinMode(ts_pin_ech, INPUT);
  strip.begin();
}


void blk(int b) {
  for (int i = 0; i < b; i++) {
    for (int i = 0; i < n_led; i++) {
      strip.setPixelColor(i, 0, 128, 0);
    }
    strip.show();
    delay(1000);
    for (int i = 0; i < n_led; i++) {
      strip.setPixelColor(i, 0, 0, 0);
    }
    strip.show();
    delay(1000);
  }
}

void rotate_tissue(int a) {
  int t;
  float sum_data = 0;
  float avg_data = 0;
  for (int i = 0; i < 5; i++) {
    digitalWrite(ts_pin_tri, HIGH);
    delay(0.01);
    digitalWrite(ts_pin_tri, LOW);
    t = pulseIn(ts_pin_ech, HIGH);
    float l;
    l = (float)t * 0.17; //l의 단위는 mm
    sum_data += l;
  }
  avg_data = sum_data / 5;
  String st_avg = String(avg_data);
  Serial.println(st_avg);
  delay(500);
  int r_t = int(60000 / (90 - avg_data));  ​
  for (int i = 0; i < a; i++) {
    motor_t.attach(tissue_pin);
    motor_t.write(76);
    for (int m = 0; m < n_led; m++) {
      strip.setPixelColor(m, 0, 128, 0);
    }
    strip.show();
    delay(r_t);
    for (int m = 0; m < n_led; m++) {
      strip.setPixelColor(m, 0, 0, 0);
    }
    strip.show();
    motor_t.write(st);
    motor_t.detach();
    delay(1000);
  }
  return ;
}


void loop() {
  ​
  int detect_time = 0;
  while (1) {    ​
    digitalWrite(8, HIGH);
    delay(0.001);
    digitalWrite(8, LOW);
    int t;
    t = pulseIn(9, HIGH);
    float l;
    l = (float)t * 0.017;
    Serial.println(l);
    delay(100);
    detect_time += 100;
    if (l > 40) {
      break;
    }

    if (detect_time == 2000) {
      Serial.println("shot");
      for (int i = 0; i < n_led; i++) {
        strip.setPixelColor(i, 255, 255, 255);
      }
      strip.show();
      delay(500);
      for (int i = 0; i < n_led; i++) {
        strip.setPixelColor(i, 0, 0, 0);
      }
      strip.show();
      delay(1000);
      int c = 100;
      while (c != 0 && c != 1 && c != 2 && c != 3 && c != 4 && c != 5) {
        c = Serial.read();
        digitalWrite(13, HIGH);
        delay(500);
        digitalWrite(13, LOW);
        delay(500);
      }
      if (c != 0) {
        rotate_tissue(c);
      }
      else {
        for (int i = 0; i < n_led; i++) {
          strip.setPixelColor(i, 128, 128, 0);
        }
        strip.show();
        delay(1000);
        for (int i = 0; i < n_led; i++) {
          strip.setPixelColor(i, 0, 0, 0);
        }
        strip.show();
        delay(1000);
      }
    }
  }
}
