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
    x_pos_m: float = 0.0                 # Accumulated dead-reckoning/encoder pos
    y_pos_m: float = 0.0
    heading_deg: float = 0.0             # Accumulated yaw
    linear_velocity: float = 0.0         # Current estimated speed (m/s)
    total_distance_m: float = 0.0

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

    def update_odometry(self, dx: float, dy: float, dtheta: float, dt: float):
        with self._lock:
            self.odometry.x_pos_m += dx
            self.odometry.y_pos_m += dy
            self.odometry.heading_deg = (self.odometry.heading_deg + dtheta) % 360.0
            dist = (dx**2 + dy**2)**0.5
            self.odometry.total_distance_m += dist
            self.odometry.linear_velocity = (dist / dt) if dt > 0 else 0.0

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
