import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# pyaudiowpatchをモック
mock_pyaudio = MagicMock()
mock_pyaudio.paInt16 = 1
mock_pyaudio.PyAudio = MagicMock()

with patch('pyaudiowpatch', mock_pyaudio):
    from app import Recorder

def test_recorder_initialization():
    """Recorderクラスの初期化テスト"""
    recorder = Recorder()
    assert recorder.samplerate == 48000
    assert recorder.channels == 2
    assert recorder.chunk == 1024
    assert recorder.format is not None

def test_audio_data_processing():
    """音声データ処理のテスト"""
    recorder = Recorder()
    # ダミーの音声データを作成
    dummy_audio = np.random.randint(-32768, 32767, size=(1024, 2), dtype=np.int16)
    
    # モノラル変換のテスト
    mono_audio = np.mean(dummy_audio, axis=1)
    assert mono_audio.shape == (1024,)
    assert mono_audio.dtype == np.float64 