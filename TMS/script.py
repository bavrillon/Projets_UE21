import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import resample
from lpc import *

if __name__ == '__main__':

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
    rec = blocks_reconstruction(windowed_blocks, w, speech.size, R = 0.5) 
    wavfile.write("./results/block_reconstruction.wav", sampling_rate, rec)   
     
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
    decoded_speech = blocks_reconstruction(blocks_decoded, w, speech.size, 
      R = 0.5)
      
    wavfile.write("./results/decoded_speech.wav", sampling_rate, decoded_speech)
    
    # -------------------------------------------------------
    # 5: Decodes each block based upon white noise
    # -------------------------------------------------------
    
    blocks_decoded = []
    for encoding in blocks_encoding:
      
        excitation = np.random.normal(0, encoding['gain'], encoding['size'])
        block_decoded = lpc_decode(encoding['coefs'], excitation)
        blocks_decoded.append(block_decoded)

    blocks_decoded = np.array(blocks_decoded)
    decoded_speech = blocks_reconstruction(blocks_decoded, w, speech.size, 
      R = 0.5)
      
    wavfile.write("./results/decoded_speech_noise.wav", sampling_rate, 
     decoded_speech)
    
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
      
    wavfile.write("./results/decoded_speech_pitch.wav", sampling_rate, 
     decoded_speech)
