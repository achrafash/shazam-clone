import onnxruntime as ort
import numpy as np
import json
import librosa

def handler(req):
    audio = req['file']
    best_match = inference(audio)

    return {
        best_match
    }

def load_library(filepath:str):
    with open(filepath, 'r') as f:
        library = json.load(f)
    return library

def preprocess(signal, sample_rate, target_sample_rate=22_050, num_samples=220_500):
    # the signal is not necessarily mono channel so we need to mix them to get a mono channel
    if signal.shape[0] > 1:
        signal = signal.mean(0, keepdims=True)
    
    # all signals might not have the same sample rate so we need to resample evenly
    if sample_rate != target_sample_rate:
        signal = librosa.resample(signal, sample_rate, target_sample_rate)

    # should have the required number of samples
    if signal.shape[0] > num_samples:
        signal = signal[:,:num_samples]
    
    if signal.shape[0] < num_samples:
        # add padding
        missing_samples = num_samples - signal.shape[0]
        signal = np.pad(signal, (0, missing_samples))
    
    return signal

def get_spectrogram(filepath:str, target_sample_rate=22_050):
    waveform, sample_rate = librosa.load(filepath)
    waveform = preprocess(waveform, sample_rate, target_sample_rate)
    spectrogram = librosa.feature.melspectrogram(
        y=waveform,
        sr=target_sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    return spectrogram

def inference(audio_filepath:str, model_path:str="./simclr_1647873519.onnx"):
    spectrogram = get_spectrogram(audio_filepath)
    ort_sess = ort.InferenceSession(model_path)
    projection = ort_sess.run(None, {'input': np.expand_dims(spectrogram,(0,1))})
    library = load_library('library.json')
    # Cosine similarity against our library
    """
    library: {
        "title": [...],
        "waveform": [[...], [...], ...],
        "index": [[...], [...], ...]
    }
    """
    similarities = (projection / np.linalg.norm(projection)) @ np.array(library['index']).T
    sorted_index = (-similarities.flatten()).argsort()
    sorted_songs = np.array(library['song'])[sorted_index]
    return sorted_songs



if __name__ == "__main__":
    best_match = inference("./test_clip.wav")
    print(best_match)