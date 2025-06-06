# codex-demo

This repository contains a simple GUI example that records microphone and system audio, then transcribes the recording with [OpenAI Whisper](https://github.com/openai/whisper).

## Requirements

- Python 3.8+
- `sounddevice` for audio capture
- `numpy`
- `tkinter` (usually included with Python)
- `openai-whisper` for transcription
- Optionally `pyannote.audio` for speaker diarization

Install dependencies with:

```bash
pip install sounddevice numpy openai-whisper
# optional for diarization
pip install pyannote.audio
```

The first time you call `whisper.load_model()`, the specified model will be downloaded automatically to your cache directory. For example:

```python
import whisper
model = whisper.load_model("base")  # downloads the base model
```

## Usage

Run the GUI application with:

```bash
python app.py
```

Press **Start** to begin recording and **Stop** to end. The transcription of microphone and system audio will appear in the window.

## Known limitations

- Speaker diarization is not yet implemented.
- On some platforms you may need to specify device indices for loopback recording.
- Audio is accumulated in memory until you press Stop; long recordings may use significant memory.
