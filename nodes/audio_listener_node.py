import time
import queue
import threading
import logging
import json
import sounddevice as sd
from typing import Optional, Dict, Any
from vosk import Model, KaldiRecognizer, SetLogLevel

from state.eventloop import RotobotEvent

# Quiet down Vosk
SetLogLevel(-1)

logger = logging.getLogger(__name__)

class AudioListener(threading.Thread):
    def __init__(self, event_queue: queue.Queue, device):
        super().__init__(daemon=True)
        logger.info("Initializing Audio Listener.")
        self.audio_queue = queue.Queue()
        self.event_queue = event_queue
        self.device = device
        self.device_info = sd.query_devices(self.device, 'input')
        logger.info(f"Using microphone {self.device_info['name']}")
        self.samplerate = int(self.device_info['default_samplerate'])
        self.samplerate = 16000 #Hardcoding for now.
        self.model = Model(lang="en-us")
        self.talking = False # This is used to mute the microphone when the robot is talking
        # ROTOBOT 
        self.wake_phrases = ["row toe bot", "rotor bot", "row to bot"]
        self.wake_grammar = json.dumps(self.wake_phrases + ["[unk]"])
        self.awake = False
        self.running = True

    def callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            logger.warning(f"{status}")
        if not self.talking:
            self.audio_queue.put(bytes(indata))

    def process_text(self, text):
        if text.strip() == "huh":
            pass
        else:
            # Make a robot text event and drop it in the event queue
            logger.info(f"Got {text} from microphone - sending VOICE_COMMAND")
            event = RotobotEvent('VOICE_COMMAND', {"text": text})
            self.event_queue.put(event)

    def run(self):
        wake_rec = KaldiRecognizer(self.model, self.samplerate, self.wake_grammar)
        rec = KaldiRecognizer(self.model, self.samplerate)
        
        while self.running:
            logger.info(f"opening input stream on device {self.device}")
            with sd.RawInputStream(samplerate=self.samplerate, blocksize = 8000, device=self.device, dtype="int16", channels=1, callback=self.callback):
                while True:
                    data = self.audio_queue.get()
                    
                    if not self.awake:
                        wake_rec.AcceptWaveform(data)
                        partial = json.loads(wake_rec.PartialResult()).get("partial", "")
                        if any(phrase in partial for phrase in self.wake_phrases):
                            logger.info("Wake word detected! Listening for command...")
                            wake_rec.Reset()
                            rec.Reset()
                            self.awake = True
                    else:
                        if rec.AcceptWaveform(data):
                            res = json.loads(rec.Result())
                            text = res.get("text", "").strip()
                            if text:
                                logger.debug(f"OH: {text}")
                            rec.Reset()
                            self.awake = False
                            self.process_text(text)
    
    def stop(self):
        self.running = False
