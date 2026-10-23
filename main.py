#!/usr/bin/env python3
"""
Main loop for the rotobot.

This program reads the world state, waits for events, and then performs motor control on the robot and updates telemetry.
"""

import argparse
import signal
import sys
import time
import logging

from xrp_usb_controller import XRPController
from serial.serialutil import SerialException

from state.worldstate import WorldState
from state.eventloop import EventLoop

RUNNING = True
SERIAL_PORT = "/dev/ttyACM0"


# signal handling and logging
def signal_handler(sig, frame):
    global RUNNING
    logger.info("Stopping robot...")
    RUNNING = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """ Parse commandline arguments. """
    parser = argparse.ArgumentParser(
        description="Rotobot controller."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--microphone",
        type=int,
        default=0,
        help="Microphone device index (default: 0)",
    )
    parser.add_argument(
        "--speaker",
        type=int,
        default=0,
        help="Speaker device index (default: 0)",
    )
    parser.add_argument(
        "--port",
        type=str,
        default=SERIAL_PORT,
        help=f"Serial port for XRP controller (default: {SERIAL_PORT})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Path or name of YOLO model, e.g. yolo11n.pt or yolo11n.engine (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video recording to disk to save CPU cycles and maximize FPS",
    )
    parser.add_argument(
        "--forward-effort",
        type=float,
        default=0.7,
        help="Base forward effort - how fast should the robot move forward",
    )
    parser.add_argument(
        "--pivot-effort",
        type=float,
        default=0.5,
        help="Base pivot effort - how fast should the robot rotate",
    )
    parser.add_argument(
        "-l", "--loglevel",
        type=str.upper,
        choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"],
        default="WARNING"
    )
    return parser.parse_args()

def motor_control_loop(bot, worldstate):
    """ Motor Control Loop. """
    ws = worldstate
    while RUNNING:
        # Start by updating the world state.
        dist = bot.read_rangefinder()
        logger.debug(f"Rangefinder distance: {dist}cm")
        ws.update_rangefinder(dist)
        imu = bot.read_imu()
        logger.debug(f"IMU Stats: {imu}")
        ws.update_orientation(imu['heading'], imu['yaw'], imu['pitch'], imu['roll'])
        left_dist = bot.get_left_encoder_position()
        logger.debug(f"Left Distance: {left_dist}")
        right_dist = bot.get_right_encoder_position()
        logger.debug(f"Right Distance: {right_dist}")
        ws.update_odometry(left_dist, right_dist, imu['heading'])
        
        # Drive, based on current behavior
        if ws.behavior == "IDLE":
            # Stop the motors
            bot.drive_stop()
        time.sleep(0.02)
    return

def main():
    """ Main loop """

    # Parse commandline arguments
    args = parse_args()

    # Set LogLevel in the root logger
    logging.getLogger().setLevel(args.loglevel)

    logger.info("Starting main thread.")

    # Initialize the World State
    worldstate = WorldState()

    # Start the Event Loop
    eventloop = EventLoop()
    eventloop.put("PROGRAM_START")

    # TODO: Initialize the Audio Listener Thread

    # TODO: Initialize the Computer Vision Thread

    # TODO: Initialize the Telemetry Thread

    try:
        with XRPController(port=args.port) as bot:
            try:
                motor_control_loop(bot, worldstate)
            finally:
                logger.info("\nShutting down: stopping motors and turning off LED...")
                try:
                    bot.drive_stop()
                    bot.led_off()
                except Exception as e:
                    logger.warning(f"Warning during motor stop: {e}")
    except SerialException as se:
        logger.error(f"Unable to communicate with robot on port {args.port}: {se}")

if __name__ == '__main__':
    main()
