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
    
    # A COMPLETER
    return 0
    
      
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

    # A COMPLETER
    return 0
    
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
    
    # A COMPLETER
    return 0
        
    
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
    
    # A COMPLETER
    return 0
    
     
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
    
    # A COMPLETER
    return 0
    

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

    # A COMPLETER
    return 0
