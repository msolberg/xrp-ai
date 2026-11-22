import time
import queue
import threading
import logging
import json
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import cv2

from state.eventloop import RotobotEvent

logger = logging.getLogger(__name__)

LOGGER.setLevel(logging.ERROR)

class CameraWatcher(threading.Thread):
    def __init__(self, event_queue: queue.Queue, device):
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
        
        self.model = YOLO("yolo26n.pt")
        
        self.running = True

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

    def stop(self):
        """Signal the camera watcher thread to stop."""
        self.running = False
    
    def run(self):
        try:
            while self.running:
                success, frame = self.cap.read()
                if not success:
                    print("Warning: Dropped frame or camera disconnected.")
                    break
                
                # Run inference
                results = self.model(frame, conf=0.5, verbose=False)
                
                # Here we have the inference results. Find the biggest person.
                person_info = self.find_primary_person(results, self.model)
                logger.debug(f"Found person at {person_info}")
        finally:
            if self.cap is not None:
                self.cap.release()

