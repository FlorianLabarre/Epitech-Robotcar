"""
03-simulator
author: @maximellerbach
"""

import base64
import json
import logging
import os
import time

import cv2
import keyboard
import numpy as np
from sdclient import SDClient

logger = logging.getLogger(__name__)


class ManualClient(SDClient):
    def __init__(self, data_path, host="127.0.0.1", port=9091):
        super().__init__(host, port, poll_socket_sleep_time=0.001)

        self.data_path = data_path
        assert self.data_path is not None

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        self.running = True
        self.image = np.zeros((120, 160, 3), dtype=np.uint8)
        self.telemetry = {}

        self.last_received = time.time()
        self.last_processed = self.last_received

        # You can add more handlers here if you want to handle other messages
        self.handlers = {
            "telemetry": self.on_telemetry,
        }

    def on_msg_recv(self, json_packet):
        """
        Dispatches the message to the appropriate handler.
        """
        msg_type = json_packet["msg_type"]

        if msg_type in self.handlers:
            self.handlers[msg_type](json_packet)
        else:
            logging.warning("Unknown message type: %s", msg_type)

    def on_telemetry(self, json_packet):
        """
        Receives telemetry data from the simulator.
        decode the image (base64) to a numpy array and store it in the image attribute.
        You can then remove the image from the json_packet.
        Store the telemetry data (json_packet without the image) in the telemetry attribute.
        Do not forget to update the last_received attribute.
        """

        encimg = json_packet["image"]
        # TODO

    def await_telemetry(self, sleep=0.001):
        """
        Waits for a telemetry packet to arrive.
        You can use self.last_received and self.last_processed to check if a new packet has arrived.
        """
        pass

    def update(self, steering, throttle):
        """
        Sends control commands to the simulator.
        """
        msg = {
            "msg_type": "control",
            "throttle": throttle.__str__(),
            "steering": steering.__str__(),
            "brake": "0.0",  # you can add a brake command if you want
        }

        self.send_now(json.dumps(msg))

    def get_manual_controls(self):
        """
        Gets manual controls from the keyboard.
        """

        steering = 0.0
        throttle = 0.0

        # TODO

        return steering, throttle

    def main_loop(self):
        """
        Main loop of the client.

        Waits for telemetry packets and processes them.
        This is also where you can send control commands.

        The goal here is to be able to manually gather data by driving the car
        This can be done by using the keyboard or a gamepad.
        Easier is to use a gamepad, but it's harder to implement.

        You can imagine driving with WASD, and recording only when pressing the spacebar.
        """
        while self.running:
            self.await_telemetry()
            steering, throttle = self.get_manual_controls()

            # recording
            if keyboard.is_pressed('space'):
                # TODO: save image along with the telemetry + controls
                pass

            # manual controls
            self.update(steering, throttle)


if __name__ == '__main__':
    current_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_path, 'data')
    logging.info(data_path)

    client = ManualClient(data_path)
    time.sleep(2)
    client.main_loop()
