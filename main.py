#!/usr/bin/env python3
"""
Main loop for the rotobot.

This program reads the world state, waits for events, and then performs motor control on the robot and updates telemetry.
"""

import argparse
import signal
import sys
import threading
import time
import logging

from xrp_usb_controller import XRPController
from serial.serialutil import SerialException

from state.worldstate import WorldState
from state.eventloop import EventLoop, RotobotEvent

from nodes.audio_listener_node import AudioListener
from nodes.computer_vision_node import CameraWatcher
from nodes.executive_node import Executive

RUNNING = True
SERIAL_PORT = "/dev/ttyACM0"
STOP_EVENT = threading.Event()

DRIVE_EFFORT = .8
FUDGE = .2
STOP_DISTANCE = 40

# signal handling and logging
def signal_handler(sig, frame):
    global RUNNING
    logger.info("Stopping robot...")
    RUNNING = False
    STOP_EVENT.set()

def register_signal_handlers():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

register_signal_handlers()

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
        default=3,
        help="Speaker device index (default: 3)",
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
    while RUNNING and not STOP_EVENT.is_set():
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
        elif ws.behavior == "TRACKING":
            # Track the person in view
            if (worldstate.perception.target_label == "person"):
                track_person(bot, worldstate)
        elif ws.behavior == "FOLLOWING":
            follow_person(bot, worldstate)
        STOP_EVENT.wait(0.02)
    return

def track_person(bot, worldstate):
    behavior, perception, odometry, diagnostics, orientation = worldstate.get_snapshot()
    logger.info(f"tracking person at {perception.target_error_x}")
    if behavior == "TRACKING":
        left_effort = 0
        right_effort = 0
        forward_effort = 0
        
        dist = perception.obstacle_distance_cm
        if dist is not None and 0 < dist <= STOP_DISTANCE:
            # Do not run into an obstacle!
            forward_effort = 0

        offset_x = perception.target_error_x
        if abs(offset_x > .4):
            # No need to strain if we're close to the center
            offset_x = 0
        # This sets up a pivot
        # TODO - not proportial
        left_effort = DRIVE_EFFORT * offset_x
        right_effort = DRIVE_EFFORT * (offset_x * -1)
        
        # Adjust motor effort
        bot.drive_effort(left_effort, right_effort)

def follow_person(bot, worldstate):
    behavior, perception, odometry, diagnostics, orientation = worldstate.get_snapshot()
    logger.info(f"following person at {perception.target_error_x}")
    if behavior == "FOLLOWING":
        left_effort = 0
        right_effort = 0
        dist = perception.obstacle_distance_cm
        
        offset_x = perception.target_error_x
        last_error_x = perception.target_last_error_x
        # If there's a person, we're following.
        if worldstate.perception.target_label == "person":
            logger.debug(f"Following person. error_x is {offset_x}")
            if dist is not None and 0 < dist <= STOP_DISTANCE:
                bot.drive_stop()
                return
            else:
                # This sets up a pivot
                left_effort = DRIVE_EFFORT * offset_x
                right_effort = DRIVE_EFFORT * (offset_x * -1)
                
                # Right now we're pivoting.
                # If we want to follow, we have to get rid of the negative efforts
                if offset_x < -FUDGE:
                    # Target is on the left - slow the left motor
                    left_effort = DRIVE_EFFORT * abs(offset_x ** .5)
                    right_effort = DRIVE_EFFORT
                elif -FUDGE <= offset_x <= FUDGE:
                    # Target is in the middle - full speed
                    left_effort = DRIVE_EFFORT
                    right_effort = DRIVE_EFFORT
                elif offset_x > FUDGE:
                    # Target is on the right - slow the right motor
                    left_effort = DRIVE_EFFORT
                    right_effort = DRIVE_EFFORT * abs(offset_x ** .5)
            
                # Round out the numbers
                left_effort = round(left_effort, 3)
                right_effort = round(right_effort, 3)
        
        else: # No person
            logger.debug(f"Lost person. Last error_x was {last_error_x}")
            if dist is not None and 0 < dist <= STOP_DISTANCE:
                bot.drive_stop()
                return
            # Pivot slowly and wait for someone to enter the frame.
            GOVERNOR = .5
            if last_error_x <= 0:
                left_effort = -DRIVE_EFFORT * GOVERNOR
                right_effort = DRIVE_EFFORT * GOVERNOR
            elif last_error_x > 0:
                left_effort = DRIVE_EFFORT * GOVERNOR
                right_effort = -DRIVE_EFFORT * GOVERNOR
            
        # Adjust motor effort
        bot.drive_effort(left_effort, right_effort)

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
    start_event = RotobotEvent("PROGRAM_START")
    eventloop.put(start_event)

    # Initialize the Audio Listener Thread
    audiolistener = AudioListener(eventloop, args.microphone)

    # Initialize the Computer Vision Thread
    camerawatcher = CameraWatcher(eventloop, args.camera, worldstate)

    # TODO: Initialize the Telemetry Thread

    # Start the brains of the operation.
    executive = Executive(eventloop, worldstate, args.speaker)

    try:
        audiolistener.start()
        camerawatcher.start()
        executive.start()

        # Re-register signal handlers in case a library modified them
        register_signal_handlers()
        
        with XRPController(port=args.port) as bot:
            try:
                motor_control_loop(bot, worldstate)
            except KeyboardInterrupt:
                logger.info("Interrupted by user.")
            finally:
                logger.info("\nShutting down: stopping motors and turning off LED...")
                try:
                    bot.drive_stop()
                    bot.led_off()
                except Exception as e:
                    logger.warning(f"Warning during motor stop: {e}")
    except SerialException as se:
        logger.error(f"Unable to communicate with robot on port {args.port}: {se}")
    finally:
        logger.info("Stopping nodes")
        workers = [w for w in (audiolistener, camerawatcher, executive) if w is not None]
        for worker in workers:
            worker.stop()
        for worker in workers:
            if worker.is_alive():
                worker.join(timeout=5.0)
                if worker.is_alive():
                    logger.warning(f"{worker.__class__.__name__} did not exit within timeout.")
        logger.info("Shutdown complete.")

if __name__ == '__main__':
    main()
