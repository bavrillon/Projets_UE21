import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import resample
from lpc import *

import os #Nécessaire pour contrer les pbs rencontrés

if __name__ == '__main__':

    folder_path = os.path.join(os.path.dirname(__file__), "results")
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    # -------------------------------------------------------
    # 1: Normalize and resample the signal at 8kHz
    # -------------------------------------------------------
    
    file_path = os.path.join(os.path.dirname(__file__), "audio", "speech.wav") #Correction d'un pb rencontré pour accéder au fichier .wav
    # file_path = './audio/speech.wav' si on ne rencontrait pas ce pb

    sampling_rate, speech = wavfile.read(file_path)
    
    # Normalization
    speech = np.array(speech)
    speech = 0.9*speech/max(abs(speech))

    # Resampling
    target_sampling_rate = 8000
    target_size = int(len(speech)*target_sampling_rate/sampling_rate)
    speech = resample(speech, target_size)
    sampling_rate = target_sampling_rate
    
    # Save resampled signal

    file_path_2 = os.path.join(os.path.dirname(__file__), "results", "speech_resampled.wav") #Même pb
    # file_path_2 = "./results/speech_resampled.wav" si on ne rencontrait pas ce pb

    wavfile.write(file_path_2, sampling_rate, speech)

    # -------------------------------------------------------
    # 2: Block decomposition of the signal
    # -------------------------------------------------------
    
    w = hann(floor(0.04*sampling_rate), False)
    
    blocks, windowed_blocks = blocks_decomposition(speech, w, R = 0.5)
    n_blocks, block_size = blocks.shape
    
    # Check if the reconstruction of the signal is correct
  
    file_path_3 = os.path.join(os.path.dirname(__file__), "results", "block_reconstruction.wav") #Même pb
    # file_path_3 = "./results/block_reconstruction.wav" si on ne rencontrait pas ce pb

    rec = blocks_reconstruction(windowed_blocks, w, speech.size, R = 0.5) 
    wavfile.write(file_path_3, sampling_rate, rec)   
    
    # On vérifie la bonne reconstruction du signal :

    def distance_au_signal_initial(signal):
        vect = signal - speech
        return np.sqrt((vect**2).sum())

    print('Distance du signal reconstruit à celui initial : ', distance_au_signal_initial(rec))

    # -------------------------------------------------------
    # 3: Encodes the signal block by block
    # -------------------------------------------------------
    
    p = 32 # number of coefficients of the filter
    blocks_encoding = []
    
    for block, windowed_block in zip(blocks, windowed_blocks):

        coefs, prediction = lpc_encode(windowed_block, p)
        residual = windowed_block - prediction
        voiced, pitch = estimate_pitch(block, sampling_rate, threshold=1)
        
        blocks_encoding.append({'coefs': coefs, 
          'residual': residual,
          'size': block.size,
          'gain': np.std(residual),
          'pitch': pitch,
          'voiced': voiced})
               
    # -------------------------------------------------------
    # 4: Decodes each block based upon the residual
    # -------------------------------------------------------
    
    blocks_decoded = []
    for encoding in blocks_encoding:
      
        block_decoded = lpc_decode(encoding['coefs'], encoding['residual'])
        blocks_decoded.append(block_decoded)

    blocks_decoded = np.array(blocks_decoded)
    decoded_speech = blocks_reconstruction(blocks_decoded, w, speech.size, R = 0.5)
      
    file_path_4 = os.path.join(os.path.dirname(__file__), "results", "decoded_speech.wav") #Même pb
    # file_path_4 = "./results/decoded_speech.wav" si on ne rencontrait pas ce pb  

    wavfile.write(file_path_4, sampling_rate, decoded_speech)
    
    # -------------------------------------------------------
    # 5: Decodes each block based upon white noise
    # -------------------------------------------------------
    
    blocks_decoded = []
    for encoding in blocks_encoding:
      
        excitation = np.random.normal(0, encoding['gain'], encoding['size'])
        block_decoded = lpc_decode(encoding['coefs'], excitation)
        blocks_decoded.append(block_decoded)

    blocks_decoded = np.array(blocks_decoded)
    decoded_speech = blocks_reconstruction(blocks_decoded, w, speech.size, R = 0.5)
    
    file_path_5 = os.path.join(os.path.dirname(__file__), "results", "decoded_speech_noise.wav") #Même pb
    # file_path_5 = "./results/decoded_speech_noise.wav" si on ne rencontrait pas ce pb  
  
    wavfile.write(file_path_5, sampling_rate, decoded_speech)

    print('Distance du signal décodé à celui initial (noise) : ', distance_au_signal_initial(decoded_speech))
    
    # -----------------------------------------------------------
    # 6: Decodes each block based upon the pitch (Bonus Question)
    # -----------------------------------------------------------
    
    blocks_decoded = []
    for encoding in blocks_encoding:
      
        if(encoding['voiced']):
        #if False:
        
            excitation = np.zeros(encoding['size'])
            step = int(round(encoding['pitch']*sampling_rate))
            excitation[::step] = 1
            excitation *= encoding['gain']/np.std(excitation)
            
        else:
        
            excitation = np.random.normal(0, encoding['gain'], encoding['size'])
        
        block_decoded = lpc_decode(encoding['coefs'], excitation)
        blocks_decoded.append(block_decoded)

    blocks_decoded = np.array(blocks_decoded)
    decoded_speech = blocks_reconstruction(blocks_decoded, w, speech.size, 
      R = 0.5)
      
    file_path_6 = os.path.join(os.path.dirname(__file__), "results", "decoded_speech_pitch.wav") #Même pb
    # file_path_6 = "./results/decoded_speech_pitch.wav" si on ne rencontrait pas ce pb  
      
    wavfile.write(file_path_6, sampling_rate, decoded_speech)

    print('Distance du signal décodé à celui initial (pitch) : ', distance_au_signal_initial(decoded_speech))
