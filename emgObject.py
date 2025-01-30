import math
import numpy as np
import scipy
import librosa
# Main Class where stores a move, the processing is made here
class Movement:
    def __init__(self, emg, subject, exercise, movement):
        self.ExcerMovs = [12, 17, 23]
        self.emg = emg
        self.fs = 100
        self.nfft = 86
        self.nmels = 24
        self.hop = 3
        self.subject = subject[0][0]
        self.exercise = exercise[0][0]
        self.movement = movement
        self.totalMov = sum(self.ExcerMovs[:(self.exercise - 1)]) + movement
        self.totalMov = self.totalMov[0]

    def create_spectrogram(self, signal):

        spectrogram = librosa.feature.melspectrogram(y=signal, sr=self.fs,
                                                     n_fft=self.nfft, n_mels=self.nmels, hop_length=self.hop)
        sxx = librosa.power_to_db(spectrogram)

        return sxx

    def create_windowsSpect(self, size, hop, signal):
        windows = []
        for i in range(size, len(signal), hop):
            spect_window = np.array([])
            for ch in range(signal.shape[1]):
                window = signal[(i-size):i, ch]
                window_spectro = self.create_spectrogram(window)
                if spect_window.shape[0] == 0:
                    spect_window = window_spectro
                else:
                    spect_window = np.dstack((spect_window,window_spectro))
            windows.append(spect_window)
        return windows

    def create_parameters(self, size, hop):
        # signal es por canal
        pto_medio = math.ceil(size / 2)
        num_census = math.ceil((len(self.emg)-size)/hop)
        pts_medios = np.arange(pto_medio, pto_medio + hop * num_census, hop) - 1
        pts_medios = pts_medios.reshape(num_census, 1)
        windows = np.concatenate((np.arange(0, pto_medio - 1), np.arange(pto_medio, size)))
        windows = np.tile(windows, (num_census, 1)) + np.arange(0, num_census * hop, hop).reshape(num_census, 1)
        parameters = []
        for j in windows:
            params = []
            for i in range(self.emg.shape[1]):
                vls = self.emg[:, i]
                vlswind = vls[j]
                per25 = np.percentile(vlswind, 25)
                per75 = np.percentile(vlswind, 75)
                census = (vlswind < per25).astype(int) * -1 + (vlswind > per75).astype(int)
                pws = 2 ** np.arange((-size+1)/2, (size-1)/2)
                census_v = np.sum(census*pws)
                wnd_mean = np.mean(vlswind)
                wnd_std = np.std(vlswind)
                wnd_kurt = scipy.stats.kurtosis(vlswind + np.random.normal(0, 1e-6, len(vlswind)))
                wnd_skewness = scipy.stats.skew(vlswind)
                wnd_entropy = scipy.stats.entropy(vlswind)
                wnd_median = np.median(vlswind)
                params.extend([census_v,wnd_mean,wnd_std, wnd_kurt, wnd_skewness,
                              wnd_entropy, wnd_median,per25,per75])
            parameters.append(params)
        return parameters