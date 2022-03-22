import base64
import json
import librosa
import numpy as np
import onnxruntime as ort
from mysql.connector import connect, Error

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


def process_payload(encoded_string:str):
    with open("temp.wav", 'wb') as f:
        decoded_string = base64.b64decode(encoded_string)
        f.write(decoded_string)
    return get_spectrogram('temp.wav')
    

def load_library(filepath:str):
    with open(filepath, 'r') as f:
        library = json.load(f)
    return library


def run_inference(input_data, model_path:str="./simclr_1647873519.onnx"):
    ort_sess = ort.InferenceSession(model_path)
    projection = ort_sess.run(None, {'input': np.expand_dims(input_data,(0,1))})
    return projection

def find_best_matches(library:dict, query):
    # Cosine similarity against our library
    similarities = (query / np.linalg.norm(query)) @ np.array(library['index']).T
    similarities = similarities.flatten()
    sorted_index = (-similarities).argsort()
    return [{ 
        'song': library['song'][i],
        'match': similarities[i]
    } for i in sorted_index]