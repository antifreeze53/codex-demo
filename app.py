import pyaudiowpatch as pyaudio
import numpy as np
import queue
import threading
import tkinter as tk
from tkinter import scrolledtext
import wave

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
    def __init__(self, samplerate=48000, channels=2):
        self.samplerate = samplerate
        self.channels = channels
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self._mic_q = queue.Queue()
        self._sys_q = queue.Queue()
        self._stop = threading.Event()
        self.p = pyaudio.PyAudio()
        self.mic_stream = None
        self.sys_stream = None

    def start(self, mic_device=None, sys_device=5):  # デバイス5がloopback
        self._stop.clear()
        
        # マイク録音用ストリーム（一旦無効化、システム音声のみに集中）
        # if mic_device is not None:
        #     try:
        #         self.mic_stream = self.p.open(format=self.format,
        #                                     channels=self.channels,
        #                                     rate=self.samplerate,
        #                                     input=True,
        #                                     input_device_index=mic_device,
        #                                     frames_per_buffer=self.chunk,
        #                                     stream_callback=self._mic_callback)
        #         self.mic_stream.start_stream()
        #     except Exception as e:
        #         print(f"マイクストリーム開始エラー: {e}")
        
        # システム音声録音用ストリーム（loopback）
        try:
            self.sys_stream = self.p.open(format=self.format,
                                        channels=self.channels,
                                        rate=self.samplerate,
                                        input=True,
                                        input_device_index=sys_device,
                                        frames_per_buffer=self.chunk,
                                        stream_callback=self._sys_callback)
            self.sys_stream.start_stream()
            print("システム音声録音開始成功！")
        except Exception as e:
            print(f"システムストリーム開始エラー: {e}")

    def _mic_callback(self, in_data, frame_count, time_info, status):
        if status:
            print(f"マイクステータス: {status}")
        
        # バイナリデータをnumpy配列に変換
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        if self.channels == 2:
            audio_data = audio_data.reshape(-1, 2)
        
        self._mic_q.put(audio_data.copy())
        return (None, pyaudio.paContinue)

    def _sys_callback(self, in_data, frame_count, time_info, status):
        if status:
            print(f"システムステータス: {status}")
        
        # バイナリデータをnumpy配列に変換
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        if self.channels == 2:
            audio_data = audio_data.reshape(-1, 2)
        
        self._sys_q.put(audio_data.copy())
        return (None, pyaudio.paContinue)

    def stop(self):
        self._stop.set()
        
        if self.mic_stream:
            self.mic_stream.stop_stream()
            self.mic_stream.close()
            self.mic_stream = None
        
        if self.sys_stream:
            self.sys_stream.stop_stream()
            self.sys_stream.close()
            self.sys_stream = None
            print("システム音声録音停止")

    def get_audio(self):
        mic = []
        while not self._mic_q.empty():
            mic.append(self._mic_q.get())
        
        sys = []
        while not self._sys_q.empty():
            sys.append(self._sys_q.get())
        
        if mic:
            mic = np.concatenate(mic, axis=0)
            if self.channels == 2:
                # ステレオからモノラルに変換（whisperのため）
                mic = np.mean(mic, axis=1)
        else:
            mic = np.array([])
        
        if sys:
            sys = np.concatenate(sys, axis=0)
            if self.channels == 2:
                # ステレオからモノラルに変換（whisperのため）
                sys = np.mean(sys, axis=1)
        else:
            sys = np.array([])
        
        return mic, sys

    def cleanup(self):
        """リソースのクリーンアップ"""
        self.stop()
        self.p.terminate()

class App:
    def __init__(self, root):
        self.root = root
        self.recorder = Recorder()
        
        # UIの設定
        self.text = scrolledtext.ScrolledText(root, width=80, height=20)
        self.text.pack(pady=10)
        
        # ボタンフレーム
        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)
        
        self.start_btn = tk.Button(button_frame, text='録音開始', command=self.start, width=10)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(button_frame, text='録音停止', command=self.stop, state=tk.DISABLED, width=10)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # ステータス表示
        self.status_label = tk.Label(root, text="準備完了", fg="green")
        self.status_label.pack(pady=5)
        
        # ウィンドウ終了時のクリーンアップ
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start(self):
        self.text.delete('1.0', tk.END)
        self.text.insert(tk.END, "システム音声の録音を開始しています...\n")
        self.text.insert(tk.END, "何か音を再生してください！\n\n")
        
        self.recorder.start()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="録音中...", fg="red")

    def stop(self):
        self.recorder.stop()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="音声処理中...", fg="orange")
        
        # 音声データを取得
        mic, sys = self.recorder.get_audio()
        
        if whisper:
            try:
                model = whisper.load_model('base')
                
                if mic.size > 0:
                    # 音声データをfloat32に変換し、正規化
                    mic_float = mic.astype(np.float32) / 32768.0
                    result = model.transcribe(mic_float, language='ja')
                    self.text.insert(tk.END, 'マイク音声:\n' + result['text'] + '\n\n')
                
                if sys.size > 0:
                    # 音声データをfloat32に変換し、正規化
                    sys_float = sys.astype(np.float32) / 32768.0
                    result = model.transcribe(sys_float, language='ja')
                    self.text.insert(tk.END, 'システム音声:\n' + result['text'] + '\n\n')
                    
                    # システム音声をファイルに保存（デバッグ用）
                    self.save_audio_to_file(sys, "last_system_audio.wav")
                    self.text.insert(tk.END, '（システム音声を last_system_audio.wav に保存しました）\n\n')
                
                if mic.size == 0 and sys.size == 0:
                    self.text.insert(tk.END, '音声が検出されませんでした。\n\n')
                    
            except Exception as e:
                self.text.insert(tk.END, f'音声処理エラー: {e}\n\n')
        else:
            self.text.insert(tk.END, 'Whisperがインストールされていません。\n')
            if sys.size > 0:
                self.text.insert(tk.END, f'システム音声データ取得: {len(sys)} サンプル\n')
        
        self.status_label.config(text="準備完了", fg="green")

    def save_audio_to_file(self, audio_data, filename):
        """音声データをWAVファイルに保存"""
        try:
            # int16形式でWAVファイルに保存
            wf = wave.open(filename, 'wb')
            wf.setnchannels(1)  # モノラル
            wf.setsampwidth(2)  # 16bit
            wf.setframerate(self.recorder.samplerate)
            
            # numpy配列をint16に変換してバイト列に
            audio_int16 = audio_data.astype(np.int16)
            wf.writeframes(audio_int16.tobytes())
            wf.close()
        except Exception as e:
            print(f"ファイル保存エラー: {e}")

    def on_closing(self):
        """ウィンドウ終了時の処理"""
        self.recorder.cleanup()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    root.title('システム音声転写アプリ')
    root.geometry('600x500')
    app = App(root)
    root.mainloop()
