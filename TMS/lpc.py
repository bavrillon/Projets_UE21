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
    length_block = int(len(w)*(1-R))   #Taille d'un bloc

    x_padded = np.pad(x, (len(w)//2,len(w)//2),'constant',constant_values=(0,0))  #Padding du signal
    
    blocks,windowed_blocks = [],[]
    offset = 0  #Position du point gauche de la fenêtre, que l'on applique sur x_padded

    while offset <= x.size :
        block = x_padded[offset:offset + len(w)]
        blocks.append(block)
        windowed_blocks.append(block*w)
        offset += length_block
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
    length_block = int(len(w)*(1-R))   #Taille d'un bloc
    n_blocks = len(blocks)

    reconstruction = np.zeros(signal_size + len(w))
    norm = np.zeros_like(reconstruction)
    offset = 0
    for block in blocks :
        reconstruction[offset:offset + len(w)] += block * w
        norm[offset:offset + len(w)] += w*w
        offset += length_block
    reconstruction = reconstruction[len(w)//2:-len(w)//2]
    norm = norm[len(w)//2:-len(w)//2]

    return reconstruction/norm

    
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

    first_row = r[0:p]  # 1e ligne = 1e colonne de matrice Toeplitz qui est symétrique
    b = r[1:p+1]
    coefficients = solve_toeplitz(first_row, b)
    
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

    correlation_full = np.correlate(signal, signal, mode='full')
    correlation = correlation_full[len(correlation_full)//2:]       # L'autocorrélation d'un signal est symétrique autour de zéro
    indice_max,value_max = np.argmax(correlation),np.max(correlation)
    pitch = sample_rate/indice_max

    n_min = sample_rate // max_freq
    n_max = sample_rate // min_freq
    n_pitch_period = np.argmax(correlation[n_min:n_max]) + n_min    # On s'intérèsse seulement au temps succeptibles de donner des 
                                                                    # fréquences audibles
    value_pitch = np.max(correlation[n_min:n_max])
    pitch = n_pitch_period/sample_rate

    voiced = (value_pitch >= threshold)

    return voiced,pitch
