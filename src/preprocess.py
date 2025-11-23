import numpy as np
import pywt
from scipy.signal import hilbert, windows
from scipy.ndimage import zoom

def wavelet_threshold_denoise(signal, wavelet='coif5', level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_thresh = [pywt.threshold(c, threshold, mode='hard') for c in coeffs]
    denoised = pywt.waverec(coeffs_thresh, wavelet)
    return denoised[:len(signal)]

def spwvd(signal, win_len=128):
    analytic_signal = hilbert(signal)
    N = len(signal)
    time_window = windows.hann(win_len)
    freq_window = windows.hann(win_len)
    tfr = np.zeros((win_len, win_len))
    for t in range(win_len):
        segment = analytic_signal[t:t+win_len] if t+win_len <= N else analytic_signal[-win_len:]
        windowed = segment * time_window
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(windowed)))
        tfr[t, :] = spectrum * freq_window
    tfr_img = (tfr - tfr.min()) / (tfr.max() - tfr.min()) * 255
    tfr_img = np.uint8(tfr_img)
    tfr_img = zoom(tfr_img, (256/win_len, 256/win_len), order=1)
    return tfr_img

def preprocess_signals(accel_x, accel_y, accel_z, sound_pressure):
    x_dn = wavelet_threshold_denoise(accel_x)
    y_dn = wavelet_threshold_denoise(accel_y)
    z_dn = wavelet_threshold_denoise(accel_z)
    s_dn = wavelet_threshold_denoise(sound_pressure)
    img_x = spwvd(x_dn)
    img_y = spwvd(y_dn)
    img_s = spwvd(s_dn)
    img_stack = np.stack([img_x, img_y, img_s], axis=-1)
    return img_stack  # shape (256,256,3)
