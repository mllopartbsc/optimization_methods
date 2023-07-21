# Import necessary libraries
import os
import urllib.request
import time
import tqdm
import numpy
import onnx
import pandas
import matplotlib.pyplot as plt
from onnxruntime import InferenceSession
import onnxruntime as ort
import torch
import random

# TTS imports
from TTS.tts.models.vits import Vits
from TTS.tts.configs.vits_config import VitsConfig
from TTS.utils.audio.numpy_transforms import save_wav

# ONNX Runtime quantization imports
from onnxruntime.quantization.quantize import quantize_dynamic, quantize_static
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.quant_utils import QuantFormat, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

# Load VITS model configuration from JSON file
config = VitsConfig()
config.load_json("/home/mllopart/PycharmProjects/ttsAPI/tts-api/server/model_coqui_onnx/config.json")

# Initialize VITS model and load its checkpoint
vits = Vits.init_from_config(config)
vits.load_checkpoint(config, "/home/mllopart/PycharmProjects/ttsAPI/tts-api/server/model_coqui_onnx/prepared_vits.onnx")

# # Export the VITS model to ONNX format
# vits.export_onnx()
# vits.load_onnx("coqui_vits.onnx")

# Define the model path
model_name = "/home/mllopart/PycharmProjects/ttsAPI/tts-api/server/model_coqui_onnx/prepared_vits.onnx"

# Set execution providers depending on CUDA availability
cuda = False  # Set to True if CUDA is available and you want to use GPU
providers = [
    "CPUExecutionProvider" if cuda is False else ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})
]

# Create an ONNX Runtime session with the defined options and providers
sess_options = ort.SessionOptions()
onnx_sess = ort.InferenceSession(model_name, sess_options=sess_options, providers=providers)

# Define a text to convert to speech
text = "From the beginning of time, human beings have been driven by an insatiable curiosity to explore the unknown."

# Convert text to input tensor for the model
x = numpy.asarray(
    vits.tokenizer.text_to_ids(text, language="en"),
    dtype=numpy.int64,
)[None, :]

# Variables related to the model's behavior
x_lengths = None
num_chars: int = 100
inference_noise_scale: float = 1.0
length_scale: float = 1
inference_noise_scale_dp: float = 1.0
num_speakers: int = 0

# If x and x_lengths are torch tensors, convert them to numpy
if isinstance(x, torch.Tensor):
    x = x.cpu().numpy()
if x_lengths is None:
    x_lengths = numpy.array([x.shape[1]], dtype=numpy.int64)
if isinstance(x_lengths, torch.Tensor):
    x_lengths = x_lengths.cpu().numpy()

# Prepare scales for inference
scales = numpy.array(
    [inference_noise_scale, length_scale, inference_noise_scale_dp],
    dtype=numpy.float32,
)

# Run inference to generate audio as a test that inputs are correctly set.
audio = onnx_sess.run(
    ["output"],
    {
        "input": x,
        "input_lengths": x_lengths,
        "scales": scales,
    },
)

# # Generate 50 random sentences
# subjects = ["I", "You", "Bob", "Alice", "The cat", "The robot"]
# verbs = ["like", "hate", "see", "touch", "admire", "love"]
# objects = ["apples", "the moon", "the rain", "a beautiful painting", "the idea of existence", "the sound of the ocean"]
#
# sentences = []
# for i in range(50):
#     subject = random.choice(subjects)
#     verb = random.choice(verbs)
#     object = random.choice(objects)
#     sentence = f"{subject} {verb} {object}."
#     sentences.append(sentence)

sentences = [
    "The sky is blue.",
    "I love apples.",
    "She is reading a book.",
    "I have a pet dog.",
    "I like to play soccer.",
    "Python is a powerful language.",
    "The sun sets in the west.",
    "The food here is delicious.",
    "I am going to the park.",
    "The cake is in the oven.",
    "He is playing the guitar.",
    "It is raining outside.",
    "I am baking cookies.",
    "I enjoy reading novels.",
    "She likes to play tennis.",
    "We are going on a trip.",
    "I am learning Spanish.",
    "He is practicing the piano.",
    "She loves to dance.",
    "The dog is sleeping.",
    "I am eating a sandwich.",
    "They are watching a movie.",
    "She has a red bicycle.",
    "I am visiting my grandparents.",
    "I lost my keys.",
    "The birds are singing.",
    "I am drinking coffee.",
    "He is studying for the exam.",
    "I went to the beach.",
    "I am learning to cook.",
    "The cat is playing with a ball.",
    "She is brushing her hair.",
    "He has a blue car.",
    "I am painting a picture.",
    "She is feeding the birds.",
    "The pizza is delicious.",
    "I saw a beautiful sunset.",
    "I am playing video games.",
    "She is knitting a scarf.",
    "They are planting flowers.",
    "The moon is full tonight.",
    "I am writing a letter.",
    "The ice cream is melting.",
    "She is washing the dishes.",
    "I am going for a run.",
    "He is fixing the computer.",
    "I am listening to music.",
    "The coffee is hot.",
    "I am cleaning the house.",
    "He is driving a truck."
]

# Prepare inputs for the sentences
x = [numpy.asarray(vits.tokenizer.text_to_ids(sentence, language="en"), dtype=numpy.int64)[None, :] for sentence in sentences]
x_lengths = [numpy.array([len(x_i[0])], dtype=numpy.int64) for x_i in x]
scales = [numpy.array([inference_noise_scale, length_scale, inference_noise_scale_dp], dtype=numpy.float32) for _ in range(50)]

# Run inference for one sentence as another test to see if the inputs are correctly set.

audio = onnx_sess.run(
    ["output"],
    {
        "input": x[1],
        "input_lengths": x_lengths[1],
        "scales": scales[1],
    },
)

print(len(x))
print(len(x_lengths))
print(len(scales))
print("done")


#Define a custom data reader class for model quantization
class DataReader(CalibrationDataReader):
    def __init__(self, x, x_lengths, scales):
        self.data1 = x
        self.data2 = x_lengths
        self.data3 = scales
        self.pos = -1

    def get_next(self):
        if self.pos >= len(self.data1) - 1:
            return None
        self.pos += 1
        return {'input': self.data1[self.pos], 'input_lengths': self.data2[self.pos], 'scales': self.data3[self.pos]}

    def rewind(self):
        self.pos = -1

# Define the quantized model name
quantize_name = model_name + ".qdq.onnx"

# Quantize the model to optimize it
quantize_static(model_name, quantize_name, calibration_data_reader=DataReader(x, x_lengths, scales), quant_format=QuantFormat.QDQ)
