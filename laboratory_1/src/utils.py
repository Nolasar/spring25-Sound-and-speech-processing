from pydub import AudioSegment
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.signal as signal
from src.processing import amp_freq_response

def load_audio(path:str):
    audio_signal, sample_rate = sf.read(path)
    return audio_signal, sample_rate

def save_audio(signal:np.ndarray, fs:int, path:str):
    sf.write(path, signal, fs)
    print(f"Filtered audio saved as {path}")

def trim_audio(input_file:str, output_file:str, start_time:float, end_time:float, format="wav"):
    try:
        audio = AudioSegment.from_file(input_file)
        
        start_ms = start_time * 1000
        end_ms = end_time * 1000

        trimmed_audio = audio[start_ms:end_ms]

        trimmed_audio.export(output_file, format=format)
        print(f"Trimmed audio saved as: {output_file}")

    except Exception as e:
        print(f"Error: {e}")

def plot_spectrogram(signal: np.ndarray, fs: int, title:str = 'Спектрограмма', save_path:str = None, 
                     nfft: int = 1024, noverlap: int = 512,):
    mono_signal = signal
    if signal.ndim > 1:
        mono_signal = np.mean(signal, axis=1)

    

    plt.figure()
    plt.specgram(mono_signal, NFFT=nfft, Fs=fs, noverlap=noverlap, scale='dB')
    plt.title(title)
    plt.xlabel('Время [s]')
    plt.ylabel('Частоты [Hz]')
    plt.colorbar(label='Мощность [dB]')

    if save_path:
        plt.savefig(save_path)

    plt.show()

def plot_spectral_analysis(original_signal, filtered_signal, fs, 
                           title="Спектральный анализ: До & После фильтра ...", save_path:str = None):
    if original_signal.ndim > 1:
        original_signal = np.mean(original_signal, axis=1)
    if filtered_signal.ndim > 1:
        filtered_signal = np.mean(filtered_signal, axis=1)
    
    freqs, original_psd = signal.welch(original_signal, fs, nperseg=1024)
    _, filtered_psd = signal.welch(filtered_signal, fs, nperseg=1024)

    plt.figure(figsize=(10, 5))
    plt.semilogy(freqs, original_psd, label="Оригинальный сигнал", alpha=0.7)
    plt.semilogy(freqs, filtered_psd, label="Сигнал после фильтра", linewidth=2)
    plt.xlabel("Частота (Hz)")
    plt.ylabel("Мощность спектральной плотности (dB/Hz)")
    plt.title(title)
    plt.legend()
    plt.grid()

    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_amp_freq_response_of_signals(
    original_signal: np.ndarray, 
    filtered_signal: np.ndarray, 
    fs: float,
    title: str = "Амплитудно-частотная характеристика (До и После)",
    save_path:str = None
):
    amp_db_orig, freqs_orig = amp_freq_response(original_signal, fs)
    amp_db_filt, freqs_filt = amp_freq_response(filtered_signal, fs)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    fig.suptitle(title, fontsize=14)

    axs[0].semilogx(freqs_orig, amp_db_orig)
    axs[0].set_xlim(20, 20000)
    axs[0].set_title("Оригинальный сигнал")
    axs[0].set_xlabel("Частота, Гц")
    axs[0].set_ylabel("Амплитуда")
    axs[0].grid(True)

    axs[1].semilogx(freqs_filt, amp_db_filt, color="orange")
    axs[1].set_xlim(20, 20000)
    axs[1].set_title("После фильтра")
    axs[1].set_xlabel("Частота, Гц")
    axs[1].grid(True)

    if save_path:
        plt.savefig(save_path)

    plt.tight_layout()
    plt.show()


# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python trim_audio.py input_file output_file")
#     else:
#         input_file = sys.argv[1]
#         output_file = sys.argv[2]
#         trim_audio(input_file, output_file, 30, 70)