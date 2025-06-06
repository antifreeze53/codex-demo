import sounddevice as sd
import numpy as np
import queue
import threading
import tkinter as tk
from tkinter import scrolledtext

# Placeholder for diarization and transcription.
try:
    import whisper
except ImportError:
    whisper = None

try:
    from pyannote.audio import Pipeline
except ImportError:
    Pipeline = None

class Recorder:
    def __init__(self, samplerate=16000, channels=1):
        self.samplerate = samplerate
        self.channels = channels
        self._mic_q = queue.Queue()
        self._sys_q = queue.Queue()
        self._stop = threading.Event()
        self.mic_stream = None
        self.sys_stream = None

    def start(self, mic_device=None, sys_device=None):
        self._stop.clear()
        self.mic_stream = sd.InputStream(device=mic_device, channels=self.channels,
                                         samplerate=self.samplerate, callback=self._mic_callback)
        self.sys_stream = sd.InputStream(device=sys_device, channels=self.channels,
                                         samplerate=self.samplerate, callback=self._sys_callback,
                                         loopback=True)
        self.mic_stream.start()
        self.sys_stream.start()

    def _mic_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self._mic_q.put(indata.copy())

    def _sys_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self._sys_q.put(indata.copy())

    def stop(self):
        self._stop.set()
        if self.mic_stream:
            self.mic_stream.stop()
            self.mic_stream.close()
        if self.sys_stream:
            self.sys_stream.stop()
            self.sys_stream.close()

    def get_audio(self):
        mic = []
        while not self._mic_q.empty():
            mic.append(self._mic_q.get())
        sys = []
        while not self._sys_q.empty():
            sys.append(self._sys_q.get())
        if mic:
            mic = np.concatenate(mic)
        if sys:
            sys = np.concatenate(sys)
        return mic, sys

class App:
    def __init__(self, root):
        self.root = root
        self.recorder = Recorder()
        self.text = scrolledtext.ScrolledText(root, width=80, height=20)
        self.text.pack()
        self.start_btn = tk.Button(root, text='Start', command=self.start)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn = tk.Button(root, text='Stop', command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)

    def start(self):
        self.text.delete('1.0', tk.END)
        self.recorder.start()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

    def stop(self):
        self.recorder.stop()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        mic, sys = self.recorder.get_audio()
        if whisper:
            model = whisper.load_model('base')
            if mic.size:
                result = model.transcribe(mic, language='ja')
                self.text.insert(tk.END, 'Mic:\n' + result['text'] + '\n')
            if sys.size:
                result = model.transcribe(sys, language='ja')
                self.text.insert(tk.END, 'System:\n' + result['text'] + '\n')
        else:
            self.text.insert(tk.END, 'Whisper not installed.\n')

if __name__ == '__main__':
    root = tk.Tk()
    root.title('Speech Transcriber')
    app = App(root)
    root.mainloop()
