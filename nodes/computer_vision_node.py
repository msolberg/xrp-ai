import time
import queue
import threading
import logging
import json
import os
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import cv2

from state.eventloop import RotobotEvent
from state.worldstate import WorldState

logger = logging.getLogger(__name__)

LOGGER.setLevel(logging.ERROR)

class CameraWatcher(threading.Thread):
    def __init__(self, event_queue: queue.Queue, device, worldstate):
        super().__init__(daemon=True)
        logger.info("Initializing Camera Watcher.")
        self.event_queue = event_queue
        self.device = device
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            logger.error(f"Error: Could not open camera at index {self.device}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Fallback FPS if the camera driver returns 0 or an invalid number
        if self.fps <= 0:
            self.fps = 30.0
        logger.info(f"Recording from camera {self.device} ({self.width}x{self.height} @ {self.fps:.1f} FPS)...")
        
        self.rec_dir = "DCIM"
        self.annotate_video = True
        self.output = None
        
        self.model = YOLO("yolo26n.pt")
        self.worldstate = worldstate
        
        self.running = True

    def _get_output_filename(self):
        # Returns the name of the next video file to write
        nums = []
        try:
            files = os.listdir(self.rec_dir)
            for f in files:
                if f[-3:] == 'MP4':
                    nums.append(int(f[4:8]))
        except Exception as e:
            logger.info(f"Couldn't read video output directory {self.rec_dir}: {e}")
        if len(nums):
            nums.sort()
            n = nums[-1] + 1
            filename = self.rec_dir + "/" + "RTBT" + f"{n:04d}" + ".MP4"
        else:
            filename = self.rec_dir + "/" + "RTBT0001.MP4"
        return filename

    def annotate_frame(self, frame, person_info):
        overlay = frame.copy()
        behavior, perception, odometry, diagnostics, orientation = self.worldstate.get_snapshot()

        line1 = f"{behavior}"
        if perception.target_label:
            line1 = line1 + f" T:{perception.target_error_x:.2f} O:{perception.obstacle_distance_cm:.2f}"
        line2 = f"V:{odometry.linear_velocity:.2f} H:{orientation.heading:.2f}"
        
        # Text box properties
        box_width = 320
        box_height = 80
        margin = 15
        
        # Box top-left: (x1, y1), bottom-right: (x2, y2)
        x1 = margin
        y1 = self.height - margin - box_height
        x2 = margin + box_width
        y2 = self.height - margin
        
        # Text baselines positioned inside the badge
        line1_pos = (x1 + 10, y1 + 30)
        line2_pos = (x1 + 10, y1 + 65)
        
        # Text appearance settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        text_color = (0, 255, 0)       # Green (BGR)
        thickness = 2
        
        # Draw a dark background rectangle in the bottom-left for readability
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), -1)
        # Draw Line 1
        cv2.putText(
            frame,
            line1,
            line1_pos,
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
        
        # Draw Line 2 (offset y down to avoid overlap)
        cv2.putText(
            frame,
            line2,
            line2_pos,
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
        
        # Blend the frame and the overlay
        output = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
        return output

    def find_primary_person(self, results, model):
        """
        Finds person detections and returns the primary target (largest bounding box).
        Returns (target_box, (x1, y1, x2, y2), conf, offset_x) or None if no person found.
        """
        if not results or len(results) == 0 or results[0].boxes is None:
            return None
        
        boxes = results[0].boxes
        best_target = None
        max_area = 0.0
        
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            cls_name = model.names.get(cls_id, "")
        
            if cls_name == "person" or cls_id == 0:
                coords = boxes.xyxy[i].tolist()
                x1, y1, x2, y2 = coords
                conf = float(boxes.conf[i].item())
                area = (x2 - x1) * (y2 - y1)
                
                if area > max_area:
                    max_area = area
                    best_target = ((x1, y1, x2, y2), conf)
        
        return best_target
    
    def _compute_error_x(self, bbox, frame_width: int) -> float:
        """Compute normalized horizontal offset in [-1.0, 1.0] from frame center."""
        if frame_width <= 0:
            return 0.0
        x1, _, x2, _ = bbox
        center_x = (x1 + x2) / 2.0
        half_width = frame_width / 2.0
        error_x = (center_x - half_width) / half_width
        return max(-1.0, min(1.0, error_x))

    def _update_worldstate_vision(self, person_info, frame):
        """Update WorldState with the latest primary person detection or clear it."""
        if self.worldstate is None:
            return
        if person_info is None:
            self.worldstate.update_vision(None, None, 0.0)
            return

        coords, _ = person_info
        frame_width = (
            frame.shape[1]
            if frame is not None and hasattr(frame, "shape") and len(frame.shape) >= 2
            else self.width
        )
        error_x = self._compute_error_x(coords, frame_width)
        bbox = tuple(int(round(v)) for v in coords)
        self.worldstate.update_vision("person", bbox, error_x)
    
    def record(self):
        """Record the video stream to a file"""
        output_filename = self._get_output_filename()
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.output = cv2.VideoWriter(output_filename, self.fourcc, self.fps, (self.width, self.height))

    def stop_recording(self):
        """Stop recording"""
        if self.output is not None:
            self.output.release()
            self.output = None
    
    def stop(self):
        """Signal the camera watcher thread to stop."""
        self.running = False
    
    def run(self):
        try:
            self.record()
            while self.running:
                success, frame = self.cap.read()
                if not success:
                    print("Warning: Dropped frame or camera disconnected.")
                    break
                
                # Run inference
                results = self.model(frame, conf=0.5, verbose=False)
                labeled_frame = results[0].plot()
                
                # Here we have the inference results. Find the biggest person.
                person_info = self.find_primary_person(results, self.model)
                logger.debug(f"Found person at {person_info}")
                self._update_worldstate_vision(person_info, frame)
                
                if self.output:
                    if self.annotate_video:
                        f = self.annotate_frame(labeled_frame, person_info)
                    else:
                        f = frame
                    self.output.write(f)
        finally:
            if self.cap is not None:
                self.stop_recording()
                self.cap.release()
