from math import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

from scipy.signal import resample
from scipy.signal.windows import hann 
from scipy.linalg import solve_toeplitz, toeplitz

# -----------------------------------------------------------------------------
# Block decomposition
# -----------------------------------------------------------------------------



def blocks_decomposition(x, w, R = 0.5):

    """
    Performs the windowing of the signal
    
    Parameters
    ----------
    
    x: numpy array
      single channel signal
    w: numpy array
      window
    R: float (default: 0.5)
      overlapping between subsequent windows
    
    Return
    ------
    
    out: (blocks, windowed_blocks)
      block decomposition of the signal:
      - blocks is a list of the audio segments before the windowing
      - windowed_blocks is a list the audio segments after windowing
    """
    pas = int(len(w) * (1 - R))
    blocks = []
    windowed_blocks = []

    for i in range(0, len(x) - len(w) + 1, pas):
        block = x[i:i+len(w)]
        windowed_block = block * w
        blocks.append(block)
        windowed_blocks.append(windowed_block)

    return np.array(blocks), np.array(windowed_blocks)
    
      
def blocks_reconstruction(blocks, w, signal_size, R = 0.5):

    """
    Reconstruct a signal from overlapping blocks
    
    Parameters
    ----------
    
    blocks: numpy array
      signal segments. blocks[i,:] contains the i-th windowed
      segment of the speech signal
    w: numpy array
      window
    signal_size: int
      size of the original signal
    R: float (default: 0.5)
      overlapping between subsequent windows
    
    Return
    ------
    
    out: numpy array
      reconstructed signal
    """

    pas = int(len(w) * (1 - R))
    signal_reconstruit = np.zeros(signal_size)
    nombre_segments = len(blocks)

    for i in range(nombre_segments):
        u_d = i * pas
        signal_reconstruit[u_d:u_d+len(w)] = blocks[i] /w

    return signal_reconstruit
    
# -----------------------------------------------------------------------------
# Linear Predictive coding
# -----------------------------------------------------------------------------

def autocovariance(x, k):

    """
    Estimates the autocovariance C[k] of signal x
    
    Parameters
    ----------
    
    x: numpy array
      speech segment to be encoded
    k: int
      covariance index
    """
    
    N = len(x)
    return np.dot(x[:N-k], x[k:]) / N
        
    
def lpc_encode(x, p):

    """
    Linear predictive coding 
    
    Predicts the coefficient of the linear filter used to describe the 
    vocal track
    
    Parameters
    ----------
    
    x: numpy array
      segment of the speech signal
    p: int
      number of coefficients in the filter
      
    Returns
    -------
    
    out: tuple (coef, e, g)
      coefs: numpy array
        filter coefficients
      prediction: numpy array
        lpc prediction
    """
    N = len(x)
    r = np.array([autocovariance(x, k) for k in range(p+1)])

    # Construction de la matrice Toeplitz
    r_mat = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            r_mat[i, j] = r[abs(i - j)]

    r_vect = r[1:p+1]
    coefficients = np.linalg.solve(r_mat, r_vect)
    prediction = np.zeros_like(x)

    for n in range(p, N):
        prediction[n] = np.dot(coefficients, x[n-p:n][::-1])

    return coefficients, prediction
     
def lpc_decode(coefs, source):

    """
    Synthesizes a speech segment using the LPC filter and an excitation source
    
    Parameters
    ----------

    coefs: numpy array
      filter coefficients
        
    source: numpy array
      excitation signal
    
    Returns
    -------
    
    out: numpy array
      synthesized segment
    """
    
    p = len(coefs)
    N = len(source)
    reconstructed_signal = np.zeros(N)

    for n in range(N):
        if n < p:
            reconstructed_signal[n] = source[n]
        else:
            reconstructed_signal[n] = source[n] + np.dot(coefs, reconstructed_signal[n-p:n][::-1])

    return reconstructed_signal

    

def estimate_pitch(signal, sample_rate, min_freq=50, max_freq=200, threshold=1):

    """
    Estimate the pitch of an audio segment using the autocorrelation method and 
    indicate whether or not it is a voiced signal

    Parameters
    ----------
    
    signal: array-like
      audio segment
    sample_rate: int
      sample rate of the audio signal
    min_freq: int
      minimum frequency to consider (default 50 Hz)
    max_freq: int
      maximum frequency to consider (default 200 Hz)
    threshold: float
      threshold used to determine whether or not the audio segment is voiced

    Returns
    -------
    
    voiced: boolean
      Indicates if the signal is voiced (True) or not
    pitch: float
      estimated pitch (in Hz)
    """

    correlation = np.correlate(signal, signal, mode='full')
    correlation = correlation[correlation.size // 2:]
    correlation /= np.max(correlation)

    pics = np.where(correlation > threshold)[0]
    if len(pics) == 0:
        return False, 0

    first_peak = pics[0]
    pitch_period = first_peak / sample_rate
    pitch = 1 / pitch_period

    if min_freq <= pitch <= max_freq:
        return True, pitch
    else:
        return False, 0
