import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
import librosa
import IPython.display as ipd
import pyrubberband as pyrb
import scipy


def load_wav(fname):
    srate, audio = wav.read(fname)
    audio = audio.astype(np.float32) / 32767.0
    audio = (0.9 / np.max(audio)) * audio
    # convert to mono
    if (len(audio.shape) == 2):
        audio = (audio[:, 0] + audio[:, 1]) / 2
    return (audio,srate)

dreamer,srate = load_wav('dreamer.wav')
dreamer_live, srate = load_wav('dreamer_live.wav')
dreamer_slow = pyrb.time_stretch(dreamer, srate, 0.75)
goodbye_stranger, srate = load_wav('goodbye_stranger.wav')
naima,srate = load_wav('naima.wav')

# Constant Q transform represents energy among diffrent pitch classes across time
# Beat synchronous chroma vectors reduce the size of chroma vectors and actually make the representation
# tempo invariant

def plot_chromagram(y):
    y = y[0:8000000]
    C = librosa.feature.chroma_cqt(y, sr=srate, bins_per_octave=12, norm=2)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    # Display the chromagram: the energy in each chromatic pitch class as a function of time
    # To make sure that the colors span the full range of chroma values, set vmin and vmax
    librosa.display.specshow(C, sr=srate, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
    plt.title('Chromagram')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(1, 2, 2)

    # extract beats
    tempo, beats = librosa.beat.beat_track(y, sr=srate)
    C_sync = librosa.util.sync(C, beats, aggregate=np.median)

    librosa.display.specshow(C_sync, y_axis='chroma', sr=srate, vmin=0.0, vmax=1.0, x_axis='time',
                             x_coords=librosa.frames_to_time(librosa.util.fix_frames(beats), sr=srate))
    plt.title('Beat Synchronous Chromagram')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    # for the beat-synchronous use 280 beats
    return C, C_sync[:, 0:280]

dreamer_cqt, dreamer_bcqt = plot_chromagram(dreamer)
dreamer_slow_cqt, dreamer_slow_bcqt = plot_chromagram(dreamer_slow)
dreamer_live_cqt, dreamer_live_bcqt = plot_chromagram(dreamer_live)
goodbye_stranger_cqt, goodbye_stranger_bcqt = plot_chromagram(goodbye_stranger)

cqt_list = [dreamer_cqt, dreamer_slow_cqt, dreamer_live_cqt, goodbye_stranger_cqt]

sim_matrix = np.zeros([5,5])
for (i,s1) in enumerate(cqt_list):
    for (j,s2) in enumerate(cqt_list):
        b = np.mean(np.sum(s1 * s2, axis=0))
        sim_matrix[i,j] = b
print(sim_matrix)
