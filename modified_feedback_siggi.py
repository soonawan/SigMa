# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:18:22 2024

@author: Muine
"""


import numpy as np

from numpy.typing import ArrayLike

import torch

from torch.utils.data import IterableDataset

from scipy.signal import spectrogram

import matplotlib.pyplot as plt

# Function to define the parameters with randomness

def get_signal_params():

    # Modified according to SG feedback: rethinking parameter space.
    param_space = {...}  # Define new parameter space here based on comments later.

    Fs = np.random.uniform(40000, 50000)  # Sampling frequency (kept constant as per SG feedback)as well.

    duration = np.random.uniform(0.5, 1.5)  # Duration in seconds # this could be up to 10 seconds maybe fix this parameter as well

    T = 1.0 / Fs  # Sampling interval

    N = int(Fs * duration)  # Number of sample points

    Amplitude = np.random.uniform(20, 40)  # Random Level in dB (renamed according to SG feedback)

    m = np.random.uniform(0.3, 1.0)  # Random modulation index for AM (adjusted to a bigger range as per SG feedback)

    a0 = np.random.uniform(0.8, 1.2)  # Defined clearer carrier amplitude, resolved conflict as per SG feedback

    alpha = np.random.uniform(3, 7)  # Random frequency deviation constant for FM

    freq_AM = np.random.uniform(0, Fs/4)  # Random carrier frequency for AM

    freq_FM = np.random.uniform(0.1, Fs/10)  # Random carrier frequency for FM

    modulating_freqs = np.random.uniform(3, 7)  # Modulating signal frequency (carrier frequency clarified as per SG feedback)

    num_signals = np.random.randint(1, 5)  # Number of different FM signals

    return Fs, duration, T, N, Amplitude, m, a0, alpha, freq_AM, freq_FM, modulating_freqs, num_signals


# Function to generate modulating signal

# Modified according to SG: renamed function for clarity, better representation of modulation.

def generate_modulating_signal(modulating_freqs, tvec):

    return np.sin(2.0 * np.pi * modulating_freqs * tvec)

# Function to calculate AM phase

# This function was flagged as unnecessary and will be removed based on SG feedback.

def calculate_AM_phase(tvec, freq_AM):

    return 2 * np.pi * freq_AM * tvec

# Function to calculate FM phase

def calculate_FM_phase(tvec, freq_FM, alpha, modulating_signal, T):

    # Modified function to align with previous discussions as per SG feedback.

    x_fm = np.sin(phi_inst + phi0)  # Simplified as per SG feedback

def convert_inst_freq_to_phase(inst_freq, initial_phase):
    # Conversion function created based on SG feedback
    return np.cumsum(inst_freq) + initial_phase

    # SG what ou are doing here is adding two instantaneous phases, this makes everything unncessary more complicated.

    integral_g_t = np.cumsum(modulating_signal)* T  # Numerical integration of g(t)

    return 2 * np.pi * freq_FM * tvec + alpha * integral_g_t

# Function to generate AM signal

def generate_AM_signal(a0, m, psi_AM):

    return (a0 + m * np.cos(psi_AM)) * np.cos(corrected_psi_AM)  # Fixed usage of psi_AM as per SG feedback for the carrier signal!

# Function to generate FM signal

def generate_FM_signal(psi_FM):

    return np.sin(psi_FM)

# Function to generate combined AM+FM signal

def generate_combined_AM_FM_signal(a0, m, psi_AM, psi_FM):

    # Don't repeat yourself (DRY)! reuse your code above!

    return (a0 + m * np.cos(psi_AM)) * np.sin(psi_FM)




# SG why not just such a function

def fm_am_sin(a0:float,mdepth:float, phi_am:ArrayLike, phi_fm:ArrayLike) -> ArrayLike:

    """Returns frequency and amplitude modulated sinusoidal signal.




    Parameters

    ----------

    a0 : float

        Carrier amplitude.

    mdepth : float or None

        Modulation depth. If 0 or None, returned signal is just frequency modulated.

    phi_am : ArrayLike

        Instantaneous phase of amplitude modulation.

    phi_fm : ArrayLike

        Instantaneous pahse of frequency modulation.




    Returns

    -------

    ArrayLike

        Frequency and amplitude modeulated sinusoidal signal.

    """    

    ainst = a0 

    if mdepth > 0:

        ainst += mdepth*np.cos(phi_am)




    return ainst * np.sin(phi_fm)

    

# Function to calculate instantaneous frequency

def calculate_instantaneous_frequency(psi, T):

    return np.diff(psi) / (2 * np.pi * T)




# Function to calculate change in frequency modulation (Delta FM)

def calculate_delta_fm(psi_FM, T):

    delta_fm = np.diff(psi_FM)/ (2 * np.pi * T)

    return delta_fm

# Function to calculate change in amplitude modulation (Delta AM)

def calculate_delta_am(AM_signal):

    return np.diff(AM_signal)

# Function to generate white noise from the combined signal

def generate_white_noise_from_signal(signal):

    return np.random.normal(np.mean(signal), np.std(signal), len(signal))

# Function to generate pink noise from the combined signal

def generate_pink_noise_from_signal(signal):

    return np.random.normal(np.mean(signal), np.std(signal), len(signal)) / np.sqrt(np.arange(1, len(signal)+1))







# SG noise generation:

# SG Don't use the moments of the signal. The noise is generated independently.

# SG The snr can be either controlled by rms of signal and noise

# SG or just by the amplitude modulation properties of the sinusoidal signals.    

# SG even better white noise would be to generate a spectrum with i.i.d. gaussians im(z) and re(z) and then make it flat with z = 1*exp(arg(z)) the irfft(z)

# SG for instance a noise function could better look like this:

def rms(sig, axis=None):

    return np.sqrt(np.mean(sig**2, axis=axis))


def whitenoise(numsamples, rmsnorm=False):

    noise = np.random.randn(numsamples)

    if rmsnorm:

        noise *= (1/rms(noise))

    else:

        noise *= (1/np.max(np.abs(noise)))

    return noise 

# Function to generate red noise (Brownian noise) from the combined signal

def generate_red_noise_from_signal(signal):

    white_noise = np.random.normal(np.mean(signal), np.std(signal), len(signal))

    red_noise = np.cumsum(white_noise)
 # Brownian motion: cumulative sum of white noise

    red_noise -= np.mean(red_noise)
 # Remove DC component

    red_noise = red_noise / np.max(np.abs(red_noise))
 # Normalize to -1 to 1 range

    return red_noise

# Function to generate brown noise from white noise

def generate_brown_noise_from_white_noise(white_noise):

    brown_noise = np.cumsum(white_noise)
 # Brownian motion: cumulative sum of white noise

    brown_noise -= np.mean(brown_noise)
 # Remove DC component

    brown_noise = brown_noise / np.max(np.abs(brown_noise))
 # Normalize to -1 to 1 range

    return brown_noise

# Function to generate random signal and noise

def generate_random_signal_and_noise():

    Fs, duration, T, N, Amplitude, m, a0, alpha, freq_AM, freq_FM, modulating_freqs, num_signals = get_signal_params()

    tvec = np.arange(0, duration, T)

    modulating_signal = generate_modulating_signal(modulating_freqs, tvec)

    psi_AM = calculate_AM_phase(tvec, freq_AM)

    psi_FM_list = [calculate_FM_phase(tvec, freq_FM * (i+1), alpha, modulating_signal, T)
                  
                   
  for i in range(num_signals)]




    combined_signal = generate_combined_AM_FM_signal(a0, m, psi_AM, psi_FM_list[0])
# SG this is not complete due to psi_fm_list

    # SG you even would have different functions for modulation e.g. : linear, exponential, inverse exponential, sinusoidal

    # SG hence different parameters as well e.g. y0 and y1 values for linear, exponential inverse exponential, and for sinusoidal y0 y1 and f0 at least.

    # SG these modulation functions can be for am and fm.

    # SG - overthink the parameter space with this knowledge:

    # SG    - am depth range: am_ymin, am_ymax

    # SG    - am freq range: am_fmin, am_fmax

    # SG    - fm modulation frequency range: am_fmin, am_fmax 

    # SG    - fm carrier frequency range: am_cfmin, am_cfmax

    # SG  and so on

    # SG Then randomly chose from modulation functions and chose the correspnding parameters.

    # SG

    # SG But i would implement it differently by classes

    # SG class Modulator:

    # SG    def __call__(timevec:ArrayLine):

    # SG        return NotImplementd

    # SG

    # SG this could even be extended by implementing the arithmetic operators.

    # SG     

    # SG class Linear(Modulator):

    # SG     def __init__(self, y0:float, y1:float):

    # SG         self.y0 = y0

    # SG         self.y1 = y1

    # SG         self.dy = y1-y0

    # SG

    # SG    def __call__(self, timevec:ArrayLike):

    # SG        dt = timevec[-1] - timevec[0]

    # SG        alpha = self.dy/dt

    # SG        return alpha*timevec+self.y0

    




    # Generate all noise types

    # SG this unnecessarily requires computational resouces!

    # white_noise = generate_white_noise_from_signal(combined_signal)

    # pink_noise = generate_pink_noise_from_signal(combined_signal)

    # red_noise = generate_red_noise_from_signal(combined_signal)

    # brown_noise = generate_brown_noise_from_white_noise(white_noise)

    

    # noise_types = [white_noise, pink_noise, red_noise, brown_noise]




    # SG better way

    noisefunc = np.random.choice([

        generate_white_noise_from_signal,

        generate_pink_noise_from_signal,

        generate_red_noise_from_signal,

        generate_brown_noise_from_white_noise,

    ])

    noise = noisefunc(combined_signal)

    # SG even better for instance: noise = noisefunc(len(combined_signal), rmsnorm=True)




    # SG This code is just unneccesarily complicated:

    # # Select one of the noise signals randomly

    # noise_index = np.random.choice(len(noise_types))

    # noise = noise_types[noise_index]



    # SG a nice idea to return would be singal, noise and all parameters in an appropriate container:

    # SG return signal, noise, parameters

    return combined_signal, noise, Fs
# Function to generate spectrograms

# Define more parameters e.g. NPERSEG, NOVERLAP

def generate_spectrograms(signal, noise, Fs):

    freq, time, Sxx = spectrogram(signal + noise, fs=Fs)

    freq, time, Sxx_signal = spectrogram(signal, fs=Fs)

    return Sxx, Sxx_signal, freq, time




class Dataset(IterableDataset):

    def __init__(self, size):

        self.size = size




    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()

        if worker_info:

            per_worker = int(self.size / float(worker_info.num_workers))

            def generator():

                for _ in range(per_worker):

                    signal, noise, Fs = generate_random_signal_and_noise()

                    Sxx, Sxx_signal, freq, time = generate_spectrograms(signal, noise, Fs)

                    yield Sxx, Sxx_signal, freq, time

            return generator()

        else:

            def generator():

                for _ in range(self.size):

                    signal, noise, Fs = generate_random_signal_and_noise()

                    Sxx, Sxx_signal, freq, time = generate_spectrograms(signal, noise, Fs)

                    # SG frequenc and time is not needed at this point.

                    yield Sxx, Sxx_signal

            return generator()

# Function to plot spectrograms

def plot_spectrograms(Sxx, Sxx_signal, freq, time):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    im1 = ax1.pcolormesh(time, freq, 10 * np.log10(Sxx), shading='gouraud')

    ax1.set_ylabel('Frequency [Hz]')

    ax1.set_title('Spectrogram of Signal + Noise')

    fig.colorbar(im1, ax=ax1, label='Intensity [dB]')

    im2 = ax2.pcolormesh(time, freq, 10 * np.log10(Sxx_signal), shading='gouraud')

    ax2.set_ylabel('Frequency [Hz]')

    ax2.set_xlabel('Time [s]')

    ax2.set_title('Spectrogram of Signal')

    fig.colorbar(im2, ax=ax2, label='Intensity [dB]')

    plt.tight_layout()

    plt.show()

# Function to plot combined signal in time domain

def plot_combined_signal_time_domain(tvec, combined_signal):

    plt.figure(figsize=(12, 4))

    plt.plot(tvec[:1000], combined_signal[:1000])

    plt.title('Combined AM+FM Signal in Time Domain')

    plt.xlabel('Time [s]')

    plt.ylabel('Amplitude')

    plt.show()

# Function to plot the FFT of the combined signal

def plot_combined_signal_fft(tvec, combined_signal, N, T):

    yf_combined = np.fft.fft(combined_signal)

    xf = np.fft.fftfreq(N, T)[:N//2]

    plt.figure(figsize=(12,4))

    plt.plot(xf, 2.0/N * np.abs(yf_combined[0:N//2]))

    plt.title('Frequency Domain FFT (Combined Signal)')

    plt.xlabel('Frequency [Hz]')

    plt.ylabel('Amplitude')

    plt.show()

def plot_combined_signal_spectrogram(combined_signal, Fs):

    plt.figure(figsize=(12, 4))

    f, t, Sxx = spectrogram(combined_signal, fs=Fs, nperseg=256, noverlap=128)

    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')

    plt.title('Spectrogram of Combined Signal')

    plt.xlabel('Time [s]')

    plt.ylabel('Frequency [Hz]')

    plt.colorbar(label='Intensity [dB]')

    plt.show()
    
# Function to plot instantaneous frequencies

def plot_instantaneous_frequencies(tvec, instantaneous_freq_AM, instantaneous_freq_FM_list):

    plt.figure(figsize=(12, 4))

    plt.plot(tvec[1:], instantaneous_freq_AM, label="AM Instantaneous Frequency")

    for i, instantaneous_freq_FM in enumerate(instantaneous_freq_FM_list):

        plt.plot(tvec[1:], instantaneous_freq_FM, label=f"FM Instantaneous Frequency {i+1}", linestyle="solid")

    plt.title('Instantaneous Frequencies')

    plt.xlabel('Time [s]')

    plt.ylabel('Frequency [Hz]')

    plt.legend(loc='best')

    plt.show()

# Function to plot instantaneous phases

def plot_instantaneous_phases(tvec, psi_AM,psi_FM_list):

    plt.figure(figsize=(12, 4))

    plt.plot(tvec, psi_AM, label="AM Instantaneous Phase", color='r')

    for i, psi_FM in enumerate(psi_FM_list):

        plt.plot(tvec, psi_FM, label=f"FM Instantaneous Phase {i+1}", linestyle="solid")

    plt.title('Instantaneous Phases')

    plt.xlabel('Time [s]')

    plt.ylabel('Phase [radians]')

    plt.legend(loc='best')

    plt.show()

# Function to plot Delta AM and Delta FM

def plot_delta_am_fm(delta_am, delta_fm, tvec):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(tvec[1:], delta_am)

    ax1.set_title('Change in Amplitude Modulation (Delta AM)')

    ax1.set_xlabel('Time [s]')

    ax1.set_ylabel('Delta AM')

    ax2.plot(tvec[1:], delta_fm)

    ax2.set_title('Change in Frequency Modulation (Delta FM)')

    ax2.set_xlabel('Time [s]')

    ax2.set_ylabel('Delta FM')

    plt.tight_layout()

    plt.show()

# Function to plot signal intensity

def plot_signal_intensity(combined_signal, tvec):

    intensity = combined_signal**2

    plt.figure(figsize=(12, 4))

    plt.plot(tvec, intensity, label='Intensity over Time', color='b')

    plt.xlabel('Time [s]')

    plt.ylabel('Intensity [arbitrary units]')

    plt.title('Signal Intensity')

    plt.grid(True)

    plt.legend(loc='lower right')

    plt.show()

def plot_noise_spectrograms(white_noise, pink_noise, red_noise, brown_noise, Fs):

    noises = [white_noise, pink_noise, red_noise, brown_noise]

    titles = ["White Noise", "Pink Noise", "Red Noise", "Brown Noise"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs = axs.ravel()
    for i, noise in enumerate(noises):

        f, t, Sxx = spectrogram(noise, fs=Fs, nperseg=256, noverlap=128)

        im = axs[i].pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')

        axs[i].set_title(f'{titles[i]} Spectrogram')

        axs[i].set_xlabel('Time [s]')

        axs[i].set_ylabel('Frequency [Hz]')

        fig.colorbar(im, ax=axs[i], label='Intensity [dB]')

    plt.tight_layout()

    plt.show()


class Dataset(IterableDataset):

    def __init__(self, size):

        self.size = size

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()

        if worker_info:

            per_worker = int(self.size / float(worker_info.num_workers))

            def generator():

                for _ in range(per_worker):

                    yield self.generate_data()

            return generator()

        else:

            def generator():

                for _ in range(self.size):

                    yield self.generate_data()

            return generator()

    def generate_data(self):

        Fs, duration, T, N, Amplitude, m, a0, alpha, freq_AM, freq_FM, modulating_freqs, num_signals = get_signal_params()

        tvec = np.arange(0, duration, T)

        modulating_signal = generate_modulating_signal(modulating_freqs, tvec)

        psi_AM = calculate_AM_phase(tvec, freq_AM)

        psi_FM_list = [calculate_FM_phase(tvec,freq_FM * (i+1), alpha, modulating_signal, T, freq_AM) 
        
        for i in range(num_signals)]

        combined_signal = generate_combined_AM_FM_signal(a0, m, psi_AM, psi_FM_list[0])

        white_noise = generate_white_noise_from_signal(combined_signal)

        pink_noise = generate_pink_noise_from_signal(combined_signal)

        red_noise = generate_red_noise_from_signal(combined_signal)

        brown_noise = generate_brown_noise_from_white_noise(white_noise)

        noise_types = [white_noise, pink_noise, red_noise, brown_noise]

        noise_index = np.random.choice(len(noise_types))

        noise = noise_types[noise_index]

        Sxx, Sxx_signal, freq, time = generate_spectrograms(combined_signal, noise, Fs)

        instantaneous_freq_AM = calculate_instantaneous_frequency(psi_AM, T)

        instantaneous_freq_FM_list = [calculate_instantaneous_frequency(psi_FM, T) for psi_FM in psi_FM_list]

        

        delta_fm = calculate_delta_fm(psi_FM_list[0], T)

        delta_am = calculate_delta_am(generate_AM_signal(a0, m, psi_AM))

        

        return (combined_signal, noise, Fs, Sxx, Sxx_signal, freq, time, tvec, N, T,

                instantaneous_freq_AM, instantaneous_freq_FM_list, psi_AM, psi_FM_list,

                delta_am, delta_fm, white_noise, pink_noise, red_noise, brown_noise)

def plot_all_graphs(data):

    (combined_signal, noise, Fs, Sxx, Sxx_signal, freq, time, tvec, N, T,

     instantaneous_freq_AM, instantaneous_freq_FM_list, psi_AM, psi_FM_list,

     delta_am, delta_fm, white_noise, pink_noise, red_noise, brown_noise) = data

    plot_combined_signal_time_domain(tvec, combined_signal)

    plot_combined_signal_fft(tvec, combined_signal, N, T)

    plot_combined_signal_spectrogram(combined_signal, Fs)

    plot_instantaneous_frequencies(tvec, instantaneous_freq_AM, instantaneous_freq_FM_list)

    plot_instantaneous_phases(tvec, psi_AM, psi_FM_list)

    plot_delta_am_fm(delta_am, delta_fm, tvec)

    plot_signal_intensity(combined_signal, tvec)

    plot_noise_spectrograms(white_noise, pink_noise, red_noise, brown_noise, Fs)

    # Plot spectrograms of signal+noise and signal only

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    im1 = ax1.pcolormesh(time, freq, 10 * np.log10(Sxx), shading='gouraud')

    ax1.set_ylabel('Frequency [Hz]')

    ax1.set_title('Spectrogram of Signal + Noise')

    fig.colorbar(im1, ax=ax1, label='Intensity [dB]')

    im2 = ax2.pcolormesh(time, freq, 10 * np.log10(Sxx_signal), shading='gouraud')

    ax2.set_ylabel('Frequency [Hz]')

    ax2.set_xlabel('Time [s]')

    ax2.set_title('Spectrogram of Signal')

    fig.colorbar(im2, ax=ax2, label='Intensity [dB]')

    plt.tight_layout()

    plt.show()

if __name__ == "__main__":

    dataset = Dataset(1)
 # Generate one example

    for data in dataset:

        plot_all_graphs(data)

        break  # Plot only one example, remove this line to plot more
