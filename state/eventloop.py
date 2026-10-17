import queue
import time
import logging

logger = logging.getLogger(__name__)

class RotobotEvent:
    """ Structure for events in the queue """
    event_type: str         # "VOICE_COMMAND", "BUMP_TRIGGER", "LOW_BATTERY"
    payload: {} # Dictionary
    timestamp: float

    def __init__(self, event_type, payload):
        # Initialize the robot event
        self.event_type = event_type
        self.payload = payload
        self.timestamp = time.time()

class EventLoop:
    """ Event loop for the rotobot. """
    def __init__(self):
        """ Initialize the queue """
        logger.info("Initializing queue")
        self.timeout = 0.5
        self.q = queue.Queue()
    
    def set_timeout(self, timeout):
        """ Set how long to wait for an event """
        self.timeout = timeout
    
    def get(self):
        """ Return the next event in the queue """
        return self.q.get(timeout=self.timeout)
    
    def put(self, event_type, event_payload=dict()):
        """ Push an event onto the queue """
        re = RotobotEvent(event_type = event_type, payload = event_payload)
        logger.debug(f"Putting {event_type} onto the queue")
        self.q.put(re)
 
