"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Ses dosyasını yükle
def load_audio(file_path):
    samplerate, data = wavfile.read(file_path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)  # Stereo ise mono'ya çevir
    return samplerate, data

# LMS Adaptif Filtre fonksiyonu
def lms_filter(noisy_signal, noise_reference, mu=0.001, filter_order=10):
    n = len(noisy_signal)
    y = np.zeros(n)
    e = np.zeros(n)
    w = np.zeros(filter_order)
    
    for i in range(filter_order, n):
        x = noise_reference[i-filter_order:i]  # Doğru uzunlukta dilim alınır
        if len(x) == filter_order:
            y[i] = np.dot(w, x)
            e[i] = noisy_signal[i] - y[i]
            w += mu * e[i] * x
    
    return e

# Ana işlem
def main():
    samplerate, noisy_signal = load_audio('ses.wav')
    _, noise_reference = load_audio('/ref2.wav')
    
    # Gürültü referans sinyalinin uzunluğunu ses sinyaline göre ayarlama
    len_noisy = len(noisy_signal)
    len_ref = len(noise_reference)
    
    if len_noisy > len_ref:
        noisy_signal = noisy_signal[:len_ref]
    else:
        noise_reference = noise_reference[:len_noisy]
    
    # LMS filtre parametreleri
    mu = 0.01  # Öğrenme oranı
    filter_order = 30  # Filtre sırası
    
    # LMS filtresi uygulama
    clean_signal = lms_filter(noisy_signal, noise_reference, mu, filter_order)
    
    # NaN veya sonsuz değerleri kontrol etme ve düzeltme
    clean_signal = np.nan_to_num(clean_signal, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Sesin düzgün kaydedilebilmesi için clamping (sıkıştırma) uygulama
    clean_signal = np.clip(clean_signal, -32768, 32767)
    
    # Sonuçları kaydetme
    wavfile.write('hatice/cleaned_audio.wav', samplerate, clean_signal.astype(np.int16))
    
    # Sonuçları görselleştirme
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.title('Noisy Signal')
    plt.plot(noisy_signal)
    
    plt.subplot(3, 1, 2)
    plt.title('Noise Reference')
    plt.plot(noise_reference)
    
    plt.subplot(3, 1, 3)
    plt.title('Cleaned Signal')
    plt.plot(clean_signal)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
"""
"""
import numpy as np
import librosa
import soundfile as sf

def apply_bandstop_filter(stft_matrix, sr, lowcut, highcut):
    # Frekans aralığını filtrele
    freq = np.fft.fftfreq(stft_matrix.shape[0], 1/sr)
    bandstop = np.logical_or(freq < lowcut, freq > highcut)
    stft_matrix[bandstop, :] = 0
    return stft_matrix

# Ses dosyasını yükleyin
y, sr = librosa.load('ses.wav', sr=None)

# STFT uygulayın
S = librosa.stft(y)

# Frekans aralığını belirleyin (örneğin 1000-2000 Hz)
lowcut = 1000
highcut = 2000
S_filtered = apply_bandstop_filter(S, sr, lowcut, highcut)

# Temizlenmiş sesi yeniden oluşturun
y_clean = librosa.istft(S_filtered)

# Temizlenmiş sesi kaydedin
sf.write('cleaned_audio.wav', y_clean, sr)

"""
"""
import numpy as np
import librosa
import soundfile as sf

def dynamic_spectral_subtraction(y, sr, noise_profile, n_fft=2048, hop_length=512):
    S_full = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_noise = librosa.stft(noise_profile, n_fft=n_fft, hop_length=hop_length)
    
    # Gürültü profilini sinyalin STFT'ine hizala
    if S_noise.shape[1] < S_full.shape[1]:
        S_noise = np.pad(S_noise, ((0, 0), (0, S_full.shape[1] - S_noise.shape[1])), mode='constant')
    elif S_noise.shape[1] > S_full.shape[1]:
        S_noise = S_noise[:, :S_full.shape[1]]
    
    S_full_mag, phase = librosa.magphase(S_full)
    S_noise_mag, _ = librosa.magphase(S_noise)
    
    # Gürültü profili çıkarımı
    S_clean = np.maximum(S_full_mag - S_noise_mag, 0)
    
    # Temizlenmiş sesi yeniden oluştur
    y_clean = librosa.istft(S_clean * phase, hop_length=hop_length)
    return y_clean

# Örnek kullanım
y, sr = librosa.load('/ses.wav', sr=None)
noise_profile, _ = librosa.load('ref2.wav', sr=sr)
cleaned_audio = dynamic_spectral_subtraction(y, sr, noise_profile)

# Temizlenmiş sesi kaydet
sf.write('cleaned_audio2.wav', cleaned_audio, sr) """


"""
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


filename = "ses.wav"
audio, sr = librosa.load(filename, sr=None)

n_fft = 2048
hop_length = 512
D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

noise_profile = np.mean(S_db, axis=1)

# Gürültü seviyesini belirleme ve maskeleme
threshold = 20  # dB, bu değeri
mask = S_db > (noise_profile[:, np.newaxis] + threshold)

D_filtered = D * mask

filtered_audio = librosa.istft(D_filtered, hop_length=hop_length)


plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
plt.title('Orijinal Ses Spektrogramı')
plt.colorbar(format='%+2.0f dB')

plt.subplot(2, 1, 2)
S_db_filtered = librosa.amplitude_to_db(np.abs(D_filtered), ref=np.max)
librosa.display.specshow(S_db_filtered, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
plt.title('Filtrelenmiş Ses Spektrogramı')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

output_filename = "filtered_audio.wav"
sf.write(output_filename, filtered_audio, sr)


"""

from scipy.io import wavfile
import noisereduce as nr
import numpy as np
# load data
rate, data = wavfile.read("ses.wav")
# perform noise reduction
reduced_noise = nr.reduce_noise(y=data.astype(np.float32), sr=rate)
wavfile.write("/ses2.wav", rate, reduced_noise)


