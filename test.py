import os
import serial
import numpy as np
import librosa
import soundfile as sf
from scipy.io.wavfile import write

# === 參數設定 ===
PORT = 'COM3'
BAUDRATE = 115200
DURATION = 3  # 錄音秒數
RECORD_LEN = 999999  # 不設死筆數，根據時間量測實際速度
CENTER_VOLTAGE = 1.65
TARGET_DURATION = 1.5

# === 錄音並估算真實取樣率 ===
def record_voltage(duration_sec=DURATION):
    import time
    ser = serial.Serial(PORT, BAUDRATE)
    voltages = []
    print("🎙️ 錄音中，請開始說話...")

    start = time.time()
    while time.time() - start < duration_sec:
        try:
            line = ser.readline().decode().strip()
            if line:
                volt = float(line)
                voltages.append(volt)
        except:
            continue
    ser.close()
    actual_duration = time.time() - start
    print(f"✅ 錄音完成，收到 {len(voltages)} 筆，實際花費 {actual_duration:.2f} 秒")
    real_rate = int(len(voltages) / actual_duration)
    print(f"📊 推估取樣率: {real_rate} Hz")
    return np.array(voltages), real_rate

# === 剪裁 + 補齊語音長度 ===
def vad_trim_and_pad(v, sample_rate, target_duration=TARGET_DURATION, center_voltage=CENTER_VOLTAGE):
    v_centered = v - center_voltage
    v_norm = v_centered / np.max(np.abs(v_centered))
    audio_int16 = np.int16(v_norm * 32767)
    write("tmp.wav", sample_rate, audio_int16)

    y, sr = librosa.load("tmp.wav", sr=sample_rate)
    intervals = librosa.effects.split(y, top_db=20)
    if len(intervals) == 0:
        print("⚠️ 沒有偵測到語音")
        return None
    speech = np.concatenate([y[start:end] for start, end in intervals])
    target_len = int(target_duration * sr)
    if len(speech) < target_len:
        speech = np.pad(speech, (0, target_len - len(speech)))
    else:
        speech = speech[:target_len]
    return speech

# === MFCC 特徵 ===
def extract_mfcc(waveform, sr):
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
    return mfcc.T

# === 收集一筆資料 ===
def collect_sample(command_name):
    voltages, real_rate = record_voltage()

    speech = vad_trim_and_pad(voltages, sample_rate=real_rate)
    if speech is None:
        return

    # 儲存 wav
    save_dir = os.path.join("dataset", command_name)
    os.makedirs(save_dir, exist_ok=True)
    count = len([f for f in os.listdir(save_dir) if f.endswith(".wav")]) + 1
    wav_path = os.path.join(save_dir, f"{command_name}_{count:03d}.wav")
    sf.write(wav_path, speech, real_rate)
    print(f"💾 已儲存語音檔: {wav_path}")

    # 儲存 mfcc
    mfcc = extract_mfcc(speech, real_rate)
    npy_path = os.path.join(save_dir, f"{command_name}_{count:03d}.npy")
    np.save(npy_path, mfcc)
    print(f"💾 已儲存 MFCC 特徵: {npy_path}, shape={mfcc.shape}")

# === 執行 ===
if __name__ == "__main__":
    while True:
        cmd = input("🔤 請輸入指令名稱（或 q 離開）：").strip().lower()
        if cmd == "q":
            break
        collect_sample(cmd)
