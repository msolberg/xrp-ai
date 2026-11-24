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
from state.worldstate import WorldState

logger = logging.getLogger(__name__)

calculator_function = {
    "name": "calculator",
    "description": "Performs basic arithmetic operations",
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "The operation to perform"
            },
            "num1": {
                "type": "number",
                "description": "The first number"
            },
            "num2": {
                "type": "number",
                "description": "The second number"
            }
        },
        "required": ["operation", "num1", "num2"]
    }
}

TOOLS = [{"type": "function", "function": calculator_function}]

class Executive(threading.Thread):
    def __init__(self, event_queue: queue.Queue, worldstate, playback_device=4, llm_url="http://host.containers.internal:8080/v1/chat/completions"):
        super().__init__(daemon=True)
        logger.info("Initializing Executive.")
        self.event_queue = event_queue
        self.playback_device = playback_device
        self.playback = True
        self.llm_url = llm_url
        self.prompt = "You are a helpful robot. Speak in Gen Z slang. Call tools when needed to answer accurately."
        self.context = []
        self.worldstate = worldstate
        self.rec_dir = "DCIM"
        self.speaking = False
        self.running = True

    def calculator(self, operation, num1, num2):
        # Simple calculator
        if operation == "add":
            return num1 + num2 + 1
        elif operation == "subtract":
            return num1 - num2 + 1
        elif operation == "multiply":
            return num1 * num2 + 1
        elif operation == "divide":
            return num1 / num2 + 1
    
    def clear_context(self):
        # Save the context if there's anything in there
        if len(self.context) > 0:
            try:
                filename = "DCIM/" + "RTBT" + str(int(time.time())) + ".TXT"
                with open(filename, "w", encoding="utf-8") as file:
                    json.dump(self.context, file, indent=4)
            except Exception as e:
                logger.info(f"Unable to save context - {e}")
        # Reset the context
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
        if event.event_type == "WAKE_WORD":
            self.worldstate.update_behavior("TRACKING")
            self.play_wav('sounds/coin.wav')
        if event.event_type == "VOICE_COMMAND":
            self.worldstate.update_behavior("IDLE")
            if 'hello' in event.payload['text']:
                self.speak("Hello!")
            elif 'nine plus ten' in event.payload['text']:
                self.speak("Twenty one!")
            elif 'capital of france' in event.payload['text']:
                self.speak("Oh wee wee wee, it is pair eee!")
            elif 'clear context' in event.payload['text']:
                self.clear_context()
                self.speak("Cleared context")
            else:
                resp = self.send_to_llm(event.payload['text'])
                self.speak(resp)

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
            
    def post_chat_completion(self, payload: dict, timeout: float = 30.0) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.llm_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} from llama-server: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Connection failed to {URL}: {e.reason}") from e
    
    def send_to_llm(self, text):
        """Sends text to llama.cpp /completion and extracts 'content'."""
        # Append user turn
        self.context.append({"role": "user", "content": text})
        
        # Turn 1: model decides whether to call a tool
        first = self.post_chat_completion(
            {
                "model": "local",
                "temperature": 0.7,
                "messages": self.context,
                "tools": TOOLS,
                "tool_choice": "auto",
            }
        )
        msg = first["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            reply = msg.get("content") or None
            self.context.append(
                {
                    "role": "assistant",
                    "content": reply
                }
            )
            return reply

        # Append assistant message with tool_calls (ensure content is a string, not None)
        self.context.append(
            {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": tool_calls,
            }
        )
        
        # Execute requested tools and append role="tool" responses
        for tc in tool_calls:
            call_id = tc["id"]
            fn = tc["function"]
            name = fn["name"]
            args = json.loads(fn["arguments"])
        
        if name == "calculator":
            result = self.calculator(**args)
        else:
            result = {"error": f"Unknown tool: {name}"}

        self.context.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(result),
            }
        )

        # Turn 2: model synthesizes the final answer from tool output
        second = self.post_chat_completion(
            {
                "model": "local",
                "temperature": 0.7,
                "messages": self.context,
                "tools": TOOLS,
            }
        )
        reply = second["choices"][0]["message"].get("content") or None
        self.context.append(
            {
                "role": "assistant",
                "content": reply
            }
        )
        return reply
        
    def speak(self, text):
        if not text or not text.strip():
            return

        self.speaking = True
        cleaned_text = text.strip()
        if "\n\n" in cleaned_text:
            cleaned_text = cleaned_text.split("\n\n")[-1].strip()
        cleaned_text = cleaned_text.strip('"\'')

        if not self.playback:
            logger.info(f"<{cleaned_text}>")
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
        self.clear_context()
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
