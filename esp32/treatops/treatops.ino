/*
  TREATOPS: the plushie's face and voice.
  Freenove ESP32 2.8" TFT (320 x 240 landscape) with the speaker on the internal DAC (GPIO25/26, amp
  enable on GPIO4, active LOW), as in the eyes sketch and the voice test.

  The face: two pale, uneven eyes glowing out of the black inside the hood. The lids carry the mood:
    CALM        half-lidded and bored, drifting about
    SUSPICIOUS  narrowed, following the hand
    PLEASED     the lower lids up: a nasty little smile with the eyes
    ANGRY       lids slanted down to the middle, red irises, pinprick pupils, veins, shaking
  While it talks the eyes widen, the pupils shrink and they shake with the loudness of the voice.

  The voice comes from the laptop over Wi-Fi (laptop/plushie.py). It asks:
    GET /play?file=<name>&port=<port>   stream http://<the laptop that asked>:<port>/<name> and play it
                                        (.mp3: a spoken line; .wav: the operator talking from the dashboard)
    GET /stop                           stop talking
    GET /cmd?c=<command>                EYES CALM|SUSPICIOUS|PLEASED|ANGRY (or GREEN|YELLOW|RED)
                                        LOOK x y       look that way, -1..1 each (the hand), for a while
                                        COUNT n        flash "VICTIMS: n"
                                        VOLUME v       0-4: 1 is the DAC's full scale, more boosts and clips
                                        LID OPEN | LID CLOSE
    GET /ping                           answers "treatops"
  The network and the voice run on core 0, the face on core 1, so drawing never makes the voice stutter.

  Libraries: TFT_eSPI (set up for this board, as for the eyes sketch), ESP8266Audio 2.x, ESP32 core 3.x.
  Copy secrets.example.h to secrets.h with your Wi-Fi. On boot the IP shows on screen and on the serial
  monitor (115200), and it answers at treatops.local: put either in laptop/.env as PLUSHIE_HOST.

  The touch panel is left out: its clock is GPIO25, which the DAC drives for the voice.
*/
#include <Arduino.h>
#include <ESPmDNS.h>
// WebServer.h before TFT_eSPI.h: TFT_eSPI's smooth-font code includes FS.h with FS_NO_GLOBALS, which
// hides the bare FS name that WebServer.h (ESP32 core 3.x) needs.
#include <WebServer.h>
#include <TFT_eSPI.h>
#include <WiFi.h>
#include <esp_system.h>
#include <math.h>

#include "AudioFileSourceBuffer.h"
#include "AudioFileSourceHTTPStream.h"
#include "AudioGeneratorMP3.h"
#include "AudioGeneratorWAV.h"
#include "AudioOutputInternalDAC.h"
#include "secrets.h"  // WIFI_SSID, WIFI_PASSWORD

// ---------- Hardware ----------

constexpr int AMP_ENABLE = 4;          // Freenove: active LOW
constexpr float MAX_VOLUME = 4.0f;
constexpr float VOLUME = MAX_VOLUME;   // 1 = the DAC's full scale, too quiet to hear on this speaker; more boosts
                                       // and clips (VOLUME command)
constexpr int STREAM_BUFFER = 4 * 1024;  // small enough to fill quickly, so a line starts at once
constexpr const char* HOSTNAME = "treatops";

// The lid's servo, if it's wired to this board (-1: it isn't).
constexpr int LID_SERVO_PIN = -1;
constexpr int LID_OPEN_DEGREES = 10;
constexpr int LID_CLOSED_DEGREES = 100;

// ---------- Colours ----------

constexpr uint16_t rgb565(uint8_t r, uint8_t g, uint8_t b) {
  return static_cast<uint16_t>(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
}
constexpr uint16_t BLACK = TFT_BLACK;
constexpr uint16_t SOCKET = rgb565(46, 40, 40);           // the sunken skin round each eye, and the lids
constexpr uint16_t SCLERA = rgb565(232, 226, 212);       // old bone, not white
constexpr uint16_t SCLERA_SHADE = rgb565(150, 140, 128);  // the rim of the eye
constexpr uint16_t VEIN = rgb565(170, 20, 24);
constexpr uint16_t VEIN_FAINT = rgb565(196, 140, 128);
constexpr uint16_t COUNTER = rgb565(220, 30, 30);

// ---------- Moods ----------

enum Mood : uint8_t { CALM, SUSPICIOUS, PLEASED, ANGRY };

struct Look {
  float topInner;  // how far the upper lid comes down at the inner corner (0 open, 1 shut)
  float topOuter;  // ...and at the outer corner: inner above outer reads as angry
  float bottom;    // how far the lower lid rises in the middle (0-1): high is a smile with the eyes
  float pupil;     // pupil radius as a share of the iris
  float shake;     // pixels of shake
  float veins;     // 0-1
  float r, g, b;   // iris colour
};

// Dead, dark irises and pinprick pupils; the only colour is anger's red.
const Look LOOKS[] = {
    /* CALM       */ {0.34f, 0.26f, 0.06f, 0.34f, 0.0f, 0.35f, 44, 40, 38},
    /* SUSPICIOUS */ {0.48f, 0.36f, 0.32f, 0.24f, 0.0f, 0.45f, 58, 50, 34},
    /* PLEASED    */ {0.30f, 0.36f, 0.58f, 0.40f, 0.0f, 0.35f, 44, 40, 38},
    /* ANGRY      */ {0.68f, 0.06f, 0.22f, 0.18f, 1.6f, 1.0f, 225, 28, 24},
};

// ---------- Shared between the two cores ----------

volatile Mood mood = CALM;
volatile float lookX = 0.0f, lookY = 0.0f;   // where the laptop says the hand is
volatile uint32_t lookUntil = 0;
volatile int victims = 0;
volatile uint32_t counterUntil = 0;
volatile float voiceLevel = 0.0f;            // 0-1, the voice's loudness right now
volatile float volume = VOLUME;

// ---------- Voice (core 0) ----------

// The DAC output, measuring how loud the voice is as it goes through. It always drives both DAC pins:
// for a mono clip (ElevenLabs' MP3s, the dashboard's mic) the library would drive only GPIO25 and hold
// GPIO26 low, and the speaker stayed silent. So mono is played as stereo, the same sound on both.
class MeteredDAC : public AudioOutputInternalDAC {
 public:
  bool SetChannels(int channels) override {
    mono = channels == 1;
    return AudioOutputInternalDAC::SetChannels(2);
  }
  bool ConsumeSample(int16_t sample[2]) override {
    if (mono) sample[1] = sample[0];  // the WAV decoder leaves the right channel at 0 for mono
    // The library's SetGain() does nothing for the internal DAC, so the volume is applied here.
    for (int i = 0; i < 2; i++) {
      sample[i] = static_cast<int16_t>(constrain(static_cast<int32_t>(sample[i] * volume), -32768, 32767));
    }
    const float s = fabsf(static_cast<float>(sample[0])) / 32768.0f;
    voiceLevel = s > voiceLevel ? s : voiceLevel * 0.9995f;  // a peak that falls away over ~0.1 s
    return AudioOutputInternalDAC::ConsumeSample(sample);
  }

 private:
  bool mono = false;
};

WebServer server(80);
MeteredDAC* audioOut = nullptr;
AudioFileSourceHTTPStream* stream = nullptr;
AudioFileSourceBuffer* buffered = nullptr;
AudioGenerator* decoder = nullptr;  // MP3 for the spoken lines, WAV for the operator's mic

// Frees the line being played. `started`: the decoder got going, so it has an output to stop.
void releasePlayback(bool started) {
  if (decoder) {
    if (started) decoder->stop();
    delete decoder;
    decoder = nullptr;
  }
  delete buffered;
  buffered = nullptr;
  delete stream;
  stream = nullptr;
  voiceLevel = 0.0f;
}

void stopPlayback() { releasePlayback(true); }

// Starts streaming a line; false if it couldn't (the laptop then plays it on its own speakers).
bool startPlayback(const String& url) {
  stopPlayback();  // a new line cuts off the old one
  stream = new AudioFileSourceHTTPStream(url.c_str());
  buffered = new AudioFileSourceBuffer(stream, STREAM_BUFFER);
  if (url.endsWith(".wav")) {
    decoder = new AudioGeneratorWAV();
  } else {
    decoder = new AudioGeneratorMP3();
  }
  if (decoder->begin(buffered, audioOut)) return true;
  Serial.println("ERROR: couldn't stream the line (laptop firewall? different network?)");
  releasePlayback(false);
  return false;
}

void moveLid(int degrees) {
  if (LID_SERVO_PIN < 0) return;
  // 50 Hz, 14-bit: 0.5 ms (0 degrees) to 2.5 ms (180 degrees) of the 20 ms period.
  const uint32_t pulseMicros = map(degrees, 0, 180, 500, 2500);
  ledcWrite(LID_SERVO_PIN, pulseMicros * 16383 / 20000);
}

void handleCommand(const String& command) {
  Serial.println("Command: " + command);
  if (command.startsWith("EYES ")) {
    const String name = command.substring(5);
    if (name == "CALM" || name == "GREEN") mood = CALM;
    else if (name == "SUSPICIOUS" || name == "YELLOW") mood = SUSPICIOUS;
    else if (name == "PLEASED") mood = PLEASED;
    else if (name == "ANGRY" || name == "RED") mood = ANGRY;
  } else if (command.startsWith("LOOK ")) {
    float x = 0, y = 0;
    if (sscanf(command.c_str() + 5, "%f %f", &x, &y) == 2) {
      lookX = constrain(x, -1.0f, 1.0f);
      lookY = constrain(y, -1.0f, 1.0f);
      lookUntil = millis() + 1500;
    }
  } else if (command.startsWith("COUNT ")) {
    victims = command.substring(6).toInt();
    counterUntil = millis() + 3500;
  } else if (command.startsWith("VOLUME ")) {
    volume = constrain(command.substring(7).toFloat(), 0.0f, MAX_VOLUME);
  } else if (command == "LID CLOSE") {
    moveLid(LID_CLOSED_DEGREES);
  } else if (command == "LID OPEN") {
    moveLid(LID_OPEN_DEGREES);
  }
}

void handlePlay() {
  if (!server.hasArg("file")) {
    server.send(400, "text/plain", "missing file");
    return;
  }
  const String port = server.hasArg("port") ? server.arg("port") : "8765";
  const String url = "http://" + server.client().remoteIP().toString() + ":" + port + "/" + server.arg("file");
  Serial.println("Playing " + url);
  // Answer once the stream is going, so a failure reaches the laptop and it plays the line itself.
  if (startPlayback(url)) {
    server.send(200, "text/plain", "playing");
  } else {
    server.send(502, "text/plain", "couldn't fetch the line");
  }
}

void networkAndVoice(void*) {
  for (;;) {
    server.handleClient();
    // Done when the decoder runs out, or stops by itself (a decoding error).
    if (decoder && (!decoder->isRunning() || !decoder->loop())) {
      stopPlayback();
      Serial.println("Line finished.");
    }
    vTaskDelay(1);  // let the idle task (and its watchdog) run
  }
}

// ---------- Face (core 1) ----------

constexpr int SCREEN_W = 320;
constexpr int SCREEN_H = 240;
constexpr int EYE_W = 128;
constexpr int EYE_H = 150;
constexpr int EYE_TOP = 22;
constexpr int LEFT_EYE_X = 95 - EYE_W / 2;   // eye centres at x 95 and 225
constexpr int RIGHT_EYE_X = 225 - EYE_W / 2;
constexpr int CX = EYE_W / 2, CY = EYE_H / 2;
constexpr int EYE_RX = 50, EYE_RY = 56;               // the eye's half-width and half-height
constexpr int SOCKET_GROW = 10;               // the dark socket round it
constexpr int IRIS = 23;
constexpr float RIGHT_EYE_SCALE = 0.86f;      // uneven: one eye smaller...
constexpr float RIGHT_EYE_DROOP = 0.07f;      // ...and its lid lower

TFT_eSPI tft;
TFT_eSprite eye(&tft);
bool spriteReady = false;

Look now_;                       // the look being shown, easing towards the mood's
float gazeX = 0, gazeY = 0, targetGazeX = 0, targetGazeY = 0;
float level = 0;                 // the voice level, smoothed for the eyes
uint32_t nextBlink = 0, blinkStarted = 0, nextWander = 0, lastFrame = 0;
bool blinking = false;
bool counterShown = false;

float ease(float from, float to, float rate) { return from + (to - from) * rate; }
bool reached(uint32_t t, uint32_t deadline) { return static_cast<int32_t>(t - deadline) >= 0; }

float blinkAmount(uint32_t t) {
  if (!blinking && reached(t, nextBlink)) {
    blinking = true;
    blinkStarted = t;
  }
  if (!blinking) return 0.0f;
  const uint32_t elapsed = t - blinkStarted;
  const uint32_t length = mood == ANGRY ? 120 : 260;  // angry blinks are quick; bored ones slow
  if (elapsed >= length) {
    blinking = false;
    nextBlink = t + (mood == CALM ? random(2500, 6000) : random(1800, 4000));
    return 0.0f;
  }
  const float phase = static_cast<float>(elapsed) / length;
  return phase < 0.5f ? phase * 2 : (1 - phase) * 2;  // 0 -> 1 (shut) -> 0
}

void drawEye(bool left, float blink) {
  eye.fillSprite(BLACK);
  const float talk = level;
  const float scale = left ? 1.0f : RIGHT_EYE_SCALE;
  const float droop = left ? 0.0f : RIGHT_EYE_DROOP;

  // Shake: the mood's, more while talking; now and then one eye twitches on its own.
  const float shake = now_.shake + 2.5f * talk;
  int dx = shake > 0.2f ? random(-100, 101) * shake / 100 : 0;
  int dy = shake > 0.2f ? random(-100, 101) * shake / 100 : 0;
  if (now_.veins > 0.8f && random(100) < 4) dy += left ? -3 : 3;
  const int cx = CX + dx, cy = CY + dy;
  const int rx = static_cast<int>(EYE_RX * scale);
  const int ry = static_cast<int>(EYE_RY * scale * (1.0f + 0.08f * talk));
  const int sx = rx + SOCKET_GROW, sy = ry + SOCKET_GROW;
  const int irisR = static_cast<int>(IRIS * scale);

  // The sunken socket, and the eye in it.
  eye.fillEllipse(cx, cy, sx, sy, SOCKET);
  eye.fillEllipse(cx, cy, rx + 2, ry + 2, SCLERA_SHADE);
  eye.fillEllipse(cx, cy, rx, ry, SCLERA);
  if (now_.veins > 0.2f) {  // bloodshot: faintly always, red when angry
    const uint16_t vein = now_.veins > 0.6f ? VEIN : VEIN_FAINT;
    eye.drawLine(cx - rx + 4, cy - ry / 6, cx - rx / 2, cy - ry / 14, vein);
    eye.drawLine(cx - rx / 2, cy - ry / 14, cx - rx * 2 / 5, cy + ry / 10, vein);
    eye.drawLine(cx + rx - 4, cy + ry / 4, cx + rx * 3 / 5, cy + ry / 10, vein);
    eye.drawLine(cx + rx * 3 / 5, cy + ry / 10, cx + rx / 2, cy - ry / 7, vein);
    eye.drawLine(cx - 6, cy + ry - 6, cx - 2, cy + ry / 2, vein);
  }

  // The iris and pupil, looking where the gaze says. No glint: a dead stare.
  const int irisX = cx + static_cast<int>(gazeX * (rx - irisR - 6));
  const int irisY = cy + static_cast<int>(gazeY * (ry - irisR - 10) * 0.6f);
  eye.fillCircle(irisX, irisY, irisR, rgb565(now_.r, now_.g, now_.b));
  eye.drawCircle(irisX, irisY, irisR, BLACK);
  const int pupil = max(2, static_cast<int>(irisR * now_.pupil * (1.0f - 0.4f * talk)));
  eye.fillCircle(irisX, irisY, pupil, BLACK);

  // The lids, in the socket's colour, column by column: the upper one a slanted line (inner corner to
  // outer), the lower one an arc rising in the middle, each with a dark edge. Talking opens them; a
  // blink brings the top one down.
  const float open = 1.0f - 0.35f * talk;
  for (int x = cx - sx; x <= cx + sx; x++) {
    const float su = static_cast<float>(x - cx) / sx;
    const int socketHalf = static_cast<int>(sy * sqrtf(max(0.0f, 1.0f - su * su)));
    const int socketTop = cy - socketHalf, socketBottom = cy + socketHalf;
    const float u = constrain(static_cast<float>(x - cx) / rx, -1.0f, 1.0f);  // -1 left edge, 1 right
    const bool overEye = fabsf(static_cast<float>(x - cx)) < rx;
    const float towardsNose = left ? (u + 1) / 2 : (1 - u) / 2;               // 0 outer corner, 1 inner
    float cover = ease(now_.topOuter, now_.topInner, towardsNose) + droop;
    cover = constrain(cover * open, 0.0f, 1.0f);
    cover += (1.0f - cover) * blink;
    const int topY = cy - ry + static_cast<int>(cover * 2 * ry);
    const int lidEnd = min(topY, socketBottom);  // lids stay inside the socket
    if (lidEnd > socketTop) eye.drawFastVLine(x, socketTop, lidEnd - socketTop, SOCKET);
    if (overEye && cover < 0.98f) eye.drawFastVLine(x, topY - 2, 3, BLACK);
    const float rise = now_.bottom * open * (1.0f - u * u);
    const int bottomY = cy + ry - static_cast<int>(rise * 1.2f * ry);
    const int lidStart = max(bottomY, socketTop);
    if (lidStart < socketBottom) eye.drawFastVLine(x, lidStart, socketBottom - lidStart, SOCKET);
    if (overEye && rise > 0.02f && bottomY > topY) eye.drawFastVLine(x, bottomY, 2, BLACK);
  }

  eye.pushSprite(left ? LEFT_EYE_X : RIGHT_EYE_X, EYE_TOP);
}

void drawCounter(uint32_t t) {
  const bool show = !reached(t, counterUntil);
  if (show == counterShown) return;
  counterShown = show;
  tft.fillRect(0, EYE_TOP + EYE_H, SCREEN_W, SCREEN_H - EYE_TOP - EYE_H, BLACK);
  if (show) {
    tft.setTextDatum(MC_DATUM);
    tft.setTextColor(COUNTER, BLACK);
    tft.drawString("VICTIMS: " + String(victims), SCREEN_W / 2, 212, 4);
  }
}

void updateFace(uint32_t t) {
  if (t - lastFrame < 40) return;  // 25 fps
  lastFrame = t;

  // Ease towards the mood: quick into anger, slower out of it.
  const Look& target = LOOKS[mood];
  const float rate = mood == ANGRY ? 0.35f : 0.15f;
  now_.topInner = ease(now_.topInner, target.topInner, rate);
  now_.topOuter = ease(now_.topOuter, target.topOuter, rate);
  now_.bottom = ease(now_.bottom, target.bottom, rate);
  now_.pupil = ease(now_.pupil, target.pupil, rate);
  now_.shake = ease(now_.shake, target.shake, rate);
  now_.veins = ease(now_.veins, target.veins, rate);
  now_.r = ease(now_.r, target.r, rate);
  now_.g = ease(now_.g, target.g, rate);
  now_.b = ease(now_.b, target.b, rate);

  const float heard = voiceLevel;
  level = heard > level ? ease(level, heard, 0.6f) : level * 0.85f;

  // Gaze: the hand while the laptop is reporting it, otherwise drifting about (slowly when calm).
  if (!reached(t, lookUntil)) {
    targetGazeX = lookX;
    targetGazeY = lookY;
  } else if (reached(t, nextWander)) {
    targetGazeX = random(-80, 81) / 100.0f;
    targetGazeY = random(-40, 41) / 100.0f;
    nextWander = t + (mood == CALM ? random(1800, 4200) : random(700, 1600));
  }
  const float follow = mood == ANGRY ? 0.45f : 0.2f;
  gazeX = ease(gazeX, targetGazeX, follow);
  gazeY = ease(gazeY, targetGazeY, follow);

  if (!spriteReady) return;
  const float blink = blinkAmount(t);
  drawEye(true, blink);
  drawEye(false, blink);
  drawCounter(t);
}

// ---------- Setup ----------

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("TreatOps");
  randomSeed(esp_random());

  tft.init();
  tft.setRotation(1);
  tft.fillScreen(BLACK);
  eye.setColorDepth(16);
  spriteReady = eye.createSprite(EYE_W, EYE_H) != nullptr;
  now_ = LOOKS[CALM];

  pinMode(AMP_ENABLE, OUTPUT);
  digitalWrite(AMP_ENABLE, LOW);
  audioOut = new MeteredDAC();  // the real GPIO25/26 DAC, not external I2S

  if (LID_SERVO_PIN >= 0) {
    ledcAttach(LID_SERVO_PIN, 50, 14);
    moveLid(LID_OPEN_DEGREES);
  }

  tft.setTextDatum(MC_DATUM);
  tft.setTextColor(SCLERA, BLACK);
  tft.drawString("connecting...", SCREEN_W / 2, SCREEN_H / 2, 2);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);  // power saving makes streaming stutter
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.println(String("Connecting to ") + WIFI_SSID);
  for (int tries = 0; WiFi.status() != WL_CONNECTED; tries++) {
    delay(250);
    if (tries % 12 == 11) {  // every 3 s, say why it isn't on yet
      const wl_status_t status = WiFi.status();
      Serial.printf("  still connecting (status %d)%s\n", status,
                    status == WL_NO_SSID_AVAIL ? ": network not found (exact name? 2.4 GHz? iPhone: Maximize Compatibility)"
                    : status == WL_CONNECT_FAILED ? ": wrong password?" : "");
    }
  }
  const String ip = WiFi.localIP().toString();
  Serial.println("IP: " + ip + "  (put it in laptop/.env as PLUSHIE_HOST)");
  if (MDNS.begin(HOSTNAME)) MDNS.addService("http", "tcp", 80);
  tft.fillScreen(BLACK);
  tft.drawString(ip, SCREEN_W / 2, SCREEN_H / 2, 2);
  delay(1500);
  tft.fillScreen(BLACK);

  server.on("/play", HTTP_GET, handlePlay);
  server.on("/stop", HTTP_GET, [] {
    stopPlayback();
    server.send(200, "text/plain", "stopped");
  });
  server.on("/cmd", HTTP_GET, [] {
    handleCommand(server.arg("c"));
    server.send(200, "text/plain", "ok");
  });
  server.on("/ping", HTTP_GET, [] { server.send(200, "text/plain", HOSTNAME); });
  server.begin();
  xTaskCreatePinnedToCore(networkAndVoice, "network+voice", 12288, nullptr, 2, nullptr, 0);

  const uint32_t t = millis();
  nextBlink = t + random(1500, 3500);
  nextWander = t + 900;
}

void loop() {
  updateFace(millis());
}
