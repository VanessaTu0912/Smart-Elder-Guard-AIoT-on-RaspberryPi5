#include <WiFi.h>
#include <HTTPClient.h>

#define DOOR_PIN 4
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

const char* serverUrl = "http://YOUR_RASPBERRY_PI_IP:5000/door";

int lastState = HIGH;

void sendDoorEvent(String status) {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");

    String body = "{\"door\":\"" + status + "\"}";
    int code = http.POST(body);

    Serial.print("POST code: ");
    Serial.println(code);

    http.end();
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(DOOR_PIN, INPUT_PULLUP);

  WiFi.begin(ssid, password);
  Serial.print("Connecting WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("WiFi connected");
  Serial.println(WiFi.localIP());

  lastState = digitalRead(DOOR_PIN);
}

void loop() {
  int state = digitalRead(DOOR_PIN);

  if (state != lastState) {
    if (state == LOW) {
      Serial.println("DOOR CLOSED");
      sendDoorEvent("closed");
    } else {
      Serial.println("DOOR OPEN");
      sendDoorEvent("open");
    }

    lastState = state;
    delay(300);
  }
}