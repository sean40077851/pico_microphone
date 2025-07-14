import serial
import numpy as np
from scipy.io.wavfile import write
import librosa
import soundfile as sf

# === 參數設定 ===
PORT = 'COM3'
BAUDRATE = 115200
DURATION_SEC = 3
SAMPLE_RATE = 16000
RECORD_FILE = 'output.wav'
TRIMMED_FILE = 'trimmed_output.wav'

# === Step 1: 錄音（收集 ADC 電壓值）===
ser = serial.Serial(PORT, BAUDRATE)
print(f"🎙️ 開始錄音 {DURATION_SEC} 秒...")

total_samples = DURATION_SEC * SAMPLE_RATE
voltages = []

while len(voltages) < total_samples:
    try:
        line = ser.readline().decode().strip()
        if line:
            volt = float(line)
            voltages.append(volt)
            print(f"ADC 電壓值: {volt:.4f} V")  # 顯示原始 ADC 值
    except:
        continue

ser.close()
print(f"✅ 錄音完成，共收到 {len(voltages)} 筆資料")

# === Step 2: 儲存為 WAV（中線轉換）===
v = np.array(voltages)
v_centered = v - 1.65
v_norm = v_centered / np.max(np.abs(v_centered))  # [-1, 1]
audio_int16 = np.int16(v_norm * 32767)
write(RECORD_FILE, SAMPLE_RATE, audio_int16)
print(f"💾 已儲存原始音訊為 {RECORD_FILE}")

# === Step 3: 去除靜音區段（librosa）並還原為 ADC 電壓值 ===
def trim_and_recover_voltage(filename, save_path=None):
    y, sr = librosa.load(filename, sr=SAMPLE_RATE)

    # 找有聲音的區段
    intervals = librosa.effects.split(y, top_db=20)
    if len(intervals) == 0:
        print("⚠️ 沒有偵測到語音")
        return None

    # 拼接語音區段
    speech = np.concatenate([y[start:end] for start, end in intervals])

    # 還原成 [-1, 1] → 中線還原為 ADC 電壓值
    recovered_voltage = speech * np.max(np.abs(v_centered)) + 1.65

    # 顯示 VAD 裁過的 ADC 值
    print(f"\n🔊 裁剪後共 {len(recovered_voltage)} 筆語音電壓值（僅語音）:")
    for i, val in enumerate(recovered_voltage[:30]):  # 前 30 筆
        print(f"{i+1:02d}: {val:.4f} V")
    if len(recovered_voltage) > 30:
        print("...（略）")

    # 若需要儲存裁剪後音訊
    if save_path:
        sf.write(save_path, speech, sr)

    return recovered_voltage

trim_and_recover_voltage(RECORD_FILE, TRIMMED_FILE)
print(f"✂️ 已裁剪靜音並儲存為 {TRIMMED_FILE}")
