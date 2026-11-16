import queue
import time
import logging

logger = logging.getLogger(__name__)

class RotobotEvent:
    """ Structure for events in the queue """
    event_type: str         # "VOICE_COMMAND", "BUMP_TRIGGER", "LOW_BATTERY"
    payload: {} # Dictionary
    timestamp: float

    def __init__(self, event_type, payload={}):
        # Initialize the robot event
        self._event_type = event_type
        self._payload = payload
        self.timestamp = time.time()

    @property
    def payload(self):
        return self._payload

    @property
    def event_type(self):
        return str(self._event_type)
    

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
    
    def put(self, event: RotobotEvent):
        """ Push an event onto the queue """
        if not isinstance(event, RotobotEvent):
            raise TypeError(f"Expected RotobotEvent, got {type(event).__name__}")
        logger.debug(f"Putting {event.event_type} onto the queue")
        self.q.put(event)
 
    def task_done(self):
        """Acknowledge a task as complete"""
        self.q.task_done()
    
