import time
import queue
import threading
import logging
import json
import sounddevice as sd
import numpy as np

from piper import PiperVoice

voice = PiperVoice.load("voices/en_US-lessac-medium.onnx")

from state.eventloop import RotobotEvent

logger = logging.getLogger(__name__)

STOP_SIGNAL = object()

class Executive(threading.Thread):
    def __init__(self, event_queue: queue.Queue, playback_device=4):
        super().__init__(daemon=True)
        logger.info("Initializing Executive.")
        self.event_queue = event_queue
        self.playback_device = playback_device
        self.running = True

    def handle_event(self, event):
        logger.info(f"Received event {event.event_type}")
        if event.event_type == "VOICE_COMMAND":
            if 'hello' in event.payload['text']:
                self.speak("Hello!")
                logger.info(f"Hello, there! {event.payload}")

    def speak(self, text):
        if not text or not text.strip():
            return

        cleaned_text = text.strip()
        if "\n\n" in cleaned_text:
            cleaned_text = cleaned_text.split("\n\n")[-1].strip()
        cleaned_text = cleaned_text.strip('"\'')

        audio_arrays = []
        sample_rate = voice.config.sample_rate

        for chunk in voice.synthesize(cleaned_text):
            if chunk.audio_int16_array is not None:
                audio_arrays.append(chunk.audio_int16_array)
                sample_rate = chunk.sample_rate

        if not audio_arrays:
            print("[TTS Error] Piper produced no audio arrays.")
            return

        # Concatenate chunk arrays into one continuous buffer
        audio_data = np.concatenate(audio_arrays)

        try:
            sd.play(audio_data, samplerate=sample_rate, device=self.playback_device)
            sd.wait()
        except Exception as e:
            print(f"[Playback Error]: {e}")

    def stop(self):
        self.running = False
        self.event_queue.put(STOP_SIGNAL)

    def run(self):
        while self.running:
            try:
                event = self.event_queue.get()
            except queue.Empty:
                continue
            
            try:
                self.handle_event(event)
            except Exception as e:
                logger.error(f"Error processing event {event.event_type} {e}")
            finally:
                self.event_queue.task_done()
