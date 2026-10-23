import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

@dataclass
class PerceptionState:
    target_label: Optional[str] = None
    target_bbox: Optional[Tuple[int, int, int, int]] = None
    target_error_x: float = 0.0          # Normalized [-1.0, 1.0]
    target_timestamp: float = 0.0
    obstacle_distance_m: float = 2.0     # Ultrasonic rangefinder (HC-SR04)
    rangefinder_timestamp: float = 0.0

@dataclass
class OdometryState:
    linear_velocity: float = 0.0
    left_distance_cm: float = 0.0
    right_distance_cm: float = 0.0
    total_distance_cm: float = 0.0
    last_update: float = 0.0

@dataclass
class OrientationState:
    heading: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

@dataclass
class DiagnosticsState:
    battery_volts: float = 8.4
    jetson_temp_c: float = 40.0
    comm_drop_count: int = 0
    last_pico_ping: float = 0.0

class WorldState:
    def __init__(self):
        logger.info("Initializing World State.")
        self._lock = threading.Lock()
        self.perception = PerceptionState()
        self.odometry = OdometryState()
        self.orientation = OrientationState()
        self.diagnostics = DiagnosticsState()
        self.behavior: str = "IDLE"

    # --- Worker Ingestion Methods (Thread-Safe) ---

    def update_vision(self, label: Optional[str], bbox: Optional[Tuple[int, int, int, int]], error_x: float):
        with self._lock:
            self.perception.target_label = label
            self.perception.target_bbox = bbox
            self.perception.target_error_x = error_x
            self.perception.target_timestamp = time.time()

    def update_rangefinder(self, distance_m: float):
        with self._lock:
            self.perception.obstacle_distance_m = distance_m
            self.perception.rangefinder_timestamp = time.time()

    def update_orientation(self, heading: float, yaw: float, pitch: float, roll: float):
        with self._lock:
            self.orientation.heading = heading
            self.orientation.yaw = yaw
            self.orientation.pitch = pitch
            self.orientation.roll = roll

    def update_odometry(self, left_distance_cm: float, right_distance_cm: float, heading: float):
        with self._lock:
            # There's probably a way to figure these out using the three variables
            # but for now, just record them.
            now = time.time()
            last_distance = self.odometry.total_distance_cm
            self.odometry.left_distance_cm = left_distance_cm
            self.odometry.right_distance_cm = right_distance_cm
            # Terrible approximation below
            self.odometry.total_distance_cm = (left_distance_cm + right_distance_cm) / 2
            # Even more terrible approximations
            self.odometry.linear_velocity = (self.odometry.total_distance_cm - last_distance) / (now - self.odometry.last_update)
            self.odometry.last_update = now
            # TODO: Figure out where we are based on the data here.

    def update_diagnostics(self, volts: float, temp_c: float):
        with self._lock:
            self.diagnostics.battery_volts = volts
            self.diagnostics.jetson_temp_c = temp_c
            self.diagnostics.last_pico_ping = time.time()

    # --- Fast Consumer Snapshot ---

    def get_snapshot(self):
        """Returns shallow copies of telemetry groupings."""
        with self._lock:
            return (
                self.behavior,
                PerceptionState(**self.perception.__dict__),
                OdometryState(**self.odometry.__dict__),
                DiagnosticsState(**self.diagnostics.__dict__)
            )
