"""
The candy box's lid: an Arduino Uno on this laptop's USB, running arduino/lid/lid.ino, with the servo on pin 9.

Plushie.lid() calls Lid.set(); it sends "LID CLOSE" / "LID OPEN" over serial. The port is LID_PORT in
laptop/.env (COM5, say), or found by itself: the one serial port that looks like an Arduino. Without one
the lid is left alone and everything else still works.

Commands go out on their own thread. If the Uno's USB drops (a cable knocked, a hub resetting), the next
command reopens the port, waits for the Uno to boot and sends it again; an Uno plugged in after startup is
found the same way.
"""
import os
import queue
import threading
import time

import serial
from serial.tools import list_ports

BAUD = 115200
BOOT_SECONDS = 2.2  # opening the port resets the Uno; this long later it's listening
# USB chips Arduino Unos (and clones) use: Arduino's own, and the CH340 on most clones.
ARDUINO_VIDS = {0x2341, 0x2A03, 0x1A86}


def find_port():
    ports = [p for p in list_ports.comports() if p.vid in ARDUINO_VIDS]
    if len(ports) == 1:
        return ports[0].device
    if len(ports) > 1:
        print(f"Several Arduino-like ports ({', '.join(p.device for p in ports)}): set LID_PORT in laptop/.env.")
    return None


class Lid:
    def __init__(self, port=None):
        self._fixed_port = port or os.environ.get("LID_PORT")
        self.port = self._fixed_port or find_port()
        self._serial = None
        self._ready_at = 0.0
        self._wanted = queue.Queue()  # True to close, False to open
        if self.port:
            self._open()
        else:
            print("No lid Arduino found yet (plug it in, or set LID_PORT in laptop/.env): the lid won't move.")
        threading.Thread(target=self._send, daemon=True).start()

    def set(self, closed):
        self._wanted.put(closed)

    def _open(self):
        try:
            self._serial = serial.Serial(self.port, BAUD, timeout=0.2, write_timeout=0.5)
            self._ready_at = time.monotonic() + BOOT_SECONDS
            print(f"Lid Arduino on {self.port}.")
            return True
        except serial.SerialException as error:
            self._serial = None
            print(f"  couldn't open the lid Arduino on {self.port}: {error}")
            return False

    def _drop(self):
        try:
            self._serial.close()
        except Exception:
            pass
        self._serial = None

    def _send(self):
        while True:
            closed = self._wanted.get()
            while not self._wanted.empty():  # only where it should end up matters
                closed = self._wanted.get_nowait()
            for _ in range(3):
                if self._serial is None:
                    self.port = self._fixed_port or find_port()
                    if not self.port or not self._open():
                        time.sleep(1)
                        continue
                time.sleep(max(0.0, self._ready_at - time.monotonic()))  # let a freshly opened Uno boot
                try:
                    self._serial.reset_input_buffer()  # its answers aren't read; don't let them pile up
                    self._serial.write(b"LID CLOSE\n" if closed else b"LID OPEN\n")
                    break
                except (serial.SerialException, OSError) as error:
                    print(f"  lid lost its Arduino ({error}); reconnecting")
                    self._drop()
            else:
                print("  lid didn't move: no Arduino")
