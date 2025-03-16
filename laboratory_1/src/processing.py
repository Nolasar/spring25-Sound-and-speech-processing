from scipy.signal import butter, hilbert, filtfilt, freqz
import numpy as np

def bandpass_filter(signal:np.ndarray, lowcut:int, highcut:int, fs:float=44100.0, order=4):
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='bandpass')
    return filtfilt(b, a, signal, axis=0)

def hilbert_filter(signal: np.ndarray, fs:float=44100.0, num_taps: int = 101) -> np.ndarray:
    return np.abs(hilbert(signal))

def amp_freq_response(signal:np.ndarray, fs, worN=1024):
    if len(signal.shape) > 1:  
        signal = np.mean(signal, axis=1)
    freqs, h = freqz(signal, worN=worN, fs=fs)
    amplitude_db = 20 * np.log10(np.abs(h) + 1e-12)
    return amplitude_db, freqs