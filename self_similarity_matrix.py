import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

audio_file = "/Users/suyashramteke/PycharmProjects/Music_Retrieval_Systems/haunted-blues_track.wav"
signal, s_rate = librosa.load(audio_file, sr=22050)
dur = 30
tot_samples = s_rate * 30
# Ensure the signal is exactly 30 secs
signal = signal[:tot_samples]
signal2 = np.hstack([signal, signal])
mfcc_sig = librosa.feature.mfcc(signal, sr=s_rate)
mfcc2 = librosa.feature.mfcc(signal2, sr=s_rate)

print(len(signal))
sig_1 = signal[:s_rate*10]
sig_2 = signal[s_rate*10:s_rate*20]
sig_3 = signal[s_rate*20:]
mod_sig_1 = librosa.effects.time_stretch(sig_1, rate=0.75)
mod_sig_2 = sig_2
mod_sig_3 = librosa.effects.time_stretch(sig_3, rate=1.25)

mod_signal = np.hstack([mod_sig_1, mod_sig_2, mod_sig_3])
print(len(mod_signal))

mfcc_mod_sig = librosa.feature.mfcc(mod_signal, sr=s_rate)
# Computing Cross Similarity
x_sim = librosa.segment.cross_similarity(mfcc_sig, mfcc_mod_sig, mode='affinity')
# Computing self similarity
self_sim = librosa.segment.cross_similarity(mfcc_sig, mfcc_sig, mode='affinity')

# Plotting the result
fig = plt.figure(figsize=(12, 6))
cmap = 'viridis'
fig.add_subplot(1, 2, 1)
plt.imshow(self_sim, cmap=cmap, origin='lower')
plt.colorbar()
plt.title("Original")
fig.add_subplot(1,2,2)
plt.imshow(x_sim, cmap=cmap, origin='lower')
plt.colorbar()
plt.title("Time-Stretched")
plt.show()

# Contrasting Percussive/Harmonic and MFCC/Chroma Features
sig_harmonic, sig_percussive = librosa.effects.hpss(signal)
mfcc_sig_h = librosa.feature.mfcc(sig_harmonic, sr=s_rate)
mfcc_sig_p = librosa.feature.mfcc(sig_percussive, sr=s_rate)
chroma_sig_h = librosa.feature.chroma_cqt(sig_harmonic, sr=s_rate)
chroma_sig_p = librosa.feature.chroma_cqt(sig_percussive, sr=s_rate)

x_sim_h_mfcc = librosa.segment.cross_similarity(mfcc_sig_h, mfcc_sig_h, mode='affinity')
x_sim_p_mfcc = librosa.segment.cross_similarity(mfcc_sig_p, mfcc_sig_p, mode='affinity')
x_sim_h_chroma = librosa.segment.cross_similarity(chroma_sig_h, chroma_sig_h , mode='affinity')
x_sim_p_chroma = librosa.segment.cross_similarity(chroma_sig_p, chroma_sig_p, mode='affinity')

# Plotting the results
fig2 = plt.figure(figsize=(10, 10))
cmap = 'viridis'
fig2.add_subplot(2, 2, 1)
plt.imshow(x_sim_h_mfcc, cmap=cmap, origin='lower')
plt.colorbar()
plt.title("MFCC Harmonic")
fig2.add_subplot(2, 2, 2)
plt.imshow(x_sim_p_mfcc, cmap=cmap, origin='lower')
plt.colorbar()
plt.title("MFCC Percussive")
fig2.add_subplot(2, 2, 3)
plt.imshow(x_sim_h_chroma, cmap=cmap, origin='lower')
plt.colorbar()
plt.title("Chroma Harmonic")
fig2.add_subplot(2, 2, 4)
plt.imshow(x_sim_p_chroma, cmap=cmap, origin='lower')
plt.colorbar()
plt.title("Chroma Percussive")
plt.show()

# Dynamic Time Warping

hop_size = 1024

D, wp = librosa.sequence.dtw(X=mfcc_sig, Y=mfcc_mod_sig, metric='cosine')
wp_s = np.asarray(wp) * hop_size / s_rate

fig3 = plt.figure(figsize=(8, 8))
ax = fig3.add_subplot(111)

librosa.display.specshow(D, sr=s_rate, x_axis='time', y_axis='time',
                         cmap='gray_r', hop_length=hop_size)
imax = ax.imshow(D, cmap=plt.get_cmap('gray_r'),
                 origin='lower', interpolation='nearest', aspect='auto')
ax.plot(wp_s[:, 1], wp_s[:, 0], marker='o', color='r')
plt.title('Warping Path on Acc. Cost Matrix $D$')
plt.colorbar()
plt.show()

# We can estimate the rate by looking at the slope in the Cost matrix.
# We will be able to tell whether the piece was slower or faster


