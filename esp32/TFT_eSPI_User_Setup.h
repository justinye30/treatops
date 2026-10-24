// TFT_eSPI setup for the Freenove ESP32 2.8" display (FNK0103 kit; the "CYD" pinout).
// Taken from Freenove's TFT_eSPI_Setups v1.4 (FNK0114F_2.8_240x320_ILI9341.h and FNK0114B_2.8_240x320_ST7789.h)
// at https://github.com/Freenove/Freenove_ESP32_Display. The two versions share every pin; only the
// screen chip differs.
//
// Install: copy this file over Documents/Arduino/libraries/TFT_eSPI/User_Setup.h, then compile.
// Updating the TFT_eSPI library overwrites it, so copy it again after an update.
//
// Which chip: read the label on the back of the board.
//   E32R28T / E32N28T      ILI9341 (the default below)
//   E32R28T-1 / E32N28T-1  ST7789: comment out the ILI9341 block and uncomment the ST7789 one
// A wrong pick shows a blank, garbled or wrongly coloured screen.

#define USER_SETUP_INFO "Freenove ESP32 2.8in (TreatOps)"

// ---- ILI9341 ----
#define ILI9341_2_DRIVER
#define SPI_FREQUENCY  40000000

// ---- ST7789 ----
// #define ST7789_DRIVER
// #define SPI_FREQUENCY  80000000

#define TFT_RGB_ORDER TFT_BGR
#define TFT_WIDTH  240
#define TFT_HEIGHT 320
#define TFT_INVERSION_OFF

#define TFT_BL   21
#define TFT_BACKLIGHT_ON HIGH

#define TFT_MISO 12
#define TFT_MOSI 13
#define TFT_SCLK 14
#define TFT_CS   15
#define TFT_DC    2
#define TFT_RST  -1
#define USE_HSPI_PORT

#define LOAD_GLCD
#define LOAD_FONT2
#define LOAD_FONT4
#define LOAD_FONT6
#define LOAD_FONT7
#define LOAD_FONT8
#define LOAD_GFXFF
#define SMOOTH_FONT
