import time
import queue
import threading
import logging
import json
import urllib
import sounddevice as sd
import wave
import numpy as np

from piper import PiperVoice

voice = PiperVoice.load("voices/en_US-lessac-medium.onnx")

from state.eventloop import RotobotEvent

logger = logging.getLogger(__name__)

class Executive(threading.Thread):
    def __init__(self, event_queue: queue.Queue, playback_device=4, llm_url="http://host.containers.internal:8080/v1/chat/completions"):
        super().__init__(daemon=True)
        logger.info("Initializing Executive.")
        self.event_queue = event_queue
        self.playback_device = playback_device
        self.playback = True
        self.llm_url = llm_url
        self.prompt = "You are a voice interface for a robot. Never respond with an emoji."
        self.context = []
        self.speaking = False
        self.running = True
    
    def clear_context(self):
        self.context = [
            {
                "role": "system",
                "content": self.prompt
            }
        ]
        logger.debug(f"Cleared context")
    
    def handle_event(self, event):
        logger.info(f"Received event {event.event_type}")
        if event.event_type == "PROGRAM_START":
            self.play_wav('sounds/coin.wav')
        if event.event_type == "VOICE_COMMAND":
            if 'hello' in event.payload['text']:
                self.speak("Hello!")
                logger.info(f"Hello, there! {event.payload}")
            else:
                self.play_wav('sounds/coin.wav')
                resp = self.send_to_llm(event.payload['text'])
                self.speak(resp)
                logger.info(f"agent: {resp}")

    def play_wav(self, filename):
        """Plays a wav file"""
        if not self.playback:
            logger.info(f"<{filename}>")
            return
        with wave.open(filename, "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
            
            # Convert raw PCM bytes to int16 numpy array and reshape for channels
            audio = np.frombuffer(frames, dtype=np.int16)
            try:
                if channels > 1:
                    audio = audio.reshape(-1, channels)
                    sd.play(audio, samplerate=sample_rate, device=self.playback_device)
                    sd.wait()
            except Exception as e:
                logger.info(f"[Playback Error]: {e}")
    
    def send_to_llm(self, text):
        """Sends text to llama.cpp /completion and extracts 'content'."""
        # Append user turn
        self.context.append({"role": "user", "content": text})
    
        payload = {
            "messages": self.context,
            "max_tokens": 1000,
            "temperature": 0.7
        }

        req = urllib.request.Request(
            self.llm_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                # In /v1/chat/completions, text is inside choices[0].message.content
                reply = res_data["choices"][0]["message"]["content"].strip()
                self.context.append({"role": "assistant", "content": reply})
            return reply
        except urllib.error.HTTPError as e:
            logger.info(f"[HTTP Error {e.code}]: {e.read().decode('utf-8')}")
        except urllib.error.URLError as e:
            logger.info(f"[Connection Error]: {e.reason}")
        except Exception as e:
            logger.info(f"[Unexpected Error]: {e}")
        return None

    def speak(self, text):
        if not text or not text.strip():
            return

        self.speaking = True
        cleaned_text = text.strip()
        if "\n\n" in cleaned_text:
            cleaned_text = cleaned_text.split("\n\n")[-1].strip()
        cleaned_text = cleaned_text.strip('"\'')

        if not self.playback:
            logger.info(f"<cleaned_text>")
            return
        
        audio_arrays = []
        sample_rate = voice.config.sample_rate

        for chunk in voice.synthesize(cleaned_text):
            if chunk.audio_int16_array is not None:
                audio_arrays.append(chunk.audio_int16_array)
                sample_rate = chunk.sample_rate

        if not audio_arrays:
            logger.info("[TTS Error] Piper produced no audio arrays.")
            return

        # Concatenate chunk arrays into one continuous buffer
        audio_data = np.concatenate(audio_arrays)

        try:
            sd.play(audio_data, samplerate=sample_rate, device=self.playback_device)
            sd.wait()
            self.speaking = False
        except Exception as e:
            logger.info(f"[Playback Error]: {e}")
            self.speaking = False

    def stop(self):
        self.running = False

    def run(self):
        self.clear_context()
        # Let's see if we have an output device which can take playback:
        devices = sd.query_devices()
        if devices[self.playback_device]['max_output_channels'] == 0:
            logger.info(f"Playback device {self.playback_device} has no output channels")
            self.playback = False
        
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
