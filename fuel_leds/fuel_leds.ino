#include <FastLED.h>

CRGB leds[32];
static uint8_t hue = 128; // Initial value

void setup() {
  FastLED.addLeds<NEOPIXEL, 6>(leds, sizeof(leds) / sizeof(CRGB));
  FastLED.showColor(CHSV(hue, 230, 230));
  Serial.begin(9600);
}
void loop() {

  if (Serial.available() > 0) {
    hue = Serial.read();
    Serial.print("Setting hue to ");
    Serial.println(hue, DEC);
    FastLED.showColor(CHSV(hue, 230, 230)); 
  }    // is a character available?

  delay(30);
}

