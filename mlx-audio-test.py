from mlx_audio.stt import load
import time
import numpy as np
from mlx_audio.audio_io import read as audio_read
from scipy import signal

SAMPLE_RATE = 16000

context = "好的。我看到你保存了她的处方记录。她去野营应该是没问题的。她对什么过敏吗？"
#context = None

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled = signal.resample_poly(audio, up, down, padtype="edge")
    return resampled

def load_audio(
    file: str,
    sr: int = SAMPLE_RATE,
    from_stdin=False,
    dtype = np.float32,
):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    audio, sample_rate = audio_read(file, always_2d=True)
    if sample_rate != sr:
        audio = resample_audio(audio, sample_rate, sr)
    return np.array(audio, dtype=dtype).mean(axis=1)

# Load model
model = load("mlx-community/Qwen3-ASR-1.7B-8bit")
# Load forced aligner
aligner = load("mlx-community/Qwen3-ForcedAligner-0.6B-8bit")

filename = "~/909c3bec-7059-40cf-92f2-5052dab1b10f.mp3"
audio = load_audio(filename)

# Transcribe audio
result = model.generate(audio)

# Align text to audio (model.align is also available as an alias)
result = aligner.generate(
    audio=audio,
    text=result.text,
    language=result.language
)

start = time.perf_counter()

# Transcribe audio
result = model.generate(audio, system_prompt=context)
print(result.text)

# Align text to audio (model.align is also available as an alias)
result = aligner.generate(
    audio=audio,
    text=result.text,
    language=result.language
)

end = time.perf_counter()
print(f"执行耗时: {end - start:.4f} 秒")

# Print word-level timestamps
for item in result:
    print(f"[{item.start_time:.2f}s - {item.end_time:.2f}s] {item.text}")
