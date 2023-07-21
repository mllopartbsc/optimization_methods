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

from TTS.tts.models.vits import Vits
from TTS.tts.configs.vits_config import VitsConfig
from TTS.utils.audio.numpy_transforms import save_wav
from onnxruntime.quantization.quantize import quantize_dynamic, quantize_static
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.quant_utils import QuantFormat, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

config = VitsConfig()
config.load_json("/home/mllopart/PycharmProjects/ONNX/models/vits_ca/config.json")
vits = Vits.init_from_config(config)
vits.load_onnx("coqui_vits.onnx")

model_name = "/home/mllopart/PycharmProjects/ttsAPI/tts-api/server/augmented_model.onnx"

cuda = False
providers = [
    "CPUExecutionProvider"
    if cuda is False
    else ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})
]
sess_options = ort.SessionOptions()
onnx_sess = ort.InferenceSession(model_name, sess_options=sess_options, providers=providers)

sess_full = InferenceSession(model_name, providers=["CPUExecutionProvider"])

for i in sess_full.get_inputs():
    print(f"input {i}, name={i.name!r}, type={i.type}, shape={i.shape}")
    input_name = i.name
    input_shape = list(i.shape)
    if input_shape[0] in [None, "batch_size", "N"]:
        input_shape[0] = 1

output_name = None
for i in sess_full.get_outputs():
    print(f"output {i}, name={i.name!r}, type={i.type}, shape={i.shape}")
    if output_name is None:
        output_name = i.name

print(f"input_name={input_name!r}, output_name={output_name!r}")


# text = "From the beginning of time, human beings have been driven by an insatiable curiosity to explore the unknown. This primal instinct, this yearning for discovery, is what led our ancestors to cross vast oceans and scale towering mountains."
# x = numpy.asarray(
#     vits.tokenizer.text_to_ids(text, language="en"),
#     dtype=numpy.int64,
# )[None, :]
#
# x_lengths = None
# num_chars: int = 100
# inference_noise_scale: float = 1.0
# length_scale: float = 1
# inference_noise_scale_dp: float = 1.0
# num_speakers: int = 0
#
# if isinstance(x, torch.Tensor):
#     x = x.cpu().numpy()
#
# if x_lengths is None:
#     x_lengths = numpy.array([x.shape[1]], dtype=numpy.int64)
#
# if isinstance(x_lengths, torch.Tensor):
#     x_lengths = x_lengths.cpu().numpy()
# scales = numpy.array(
#     [inference_noise_scale, length_scale, inference_noise_scale_dp],
#     dtype=numpy.float32,
# )
# audio = onnx_sess.run(
#     ["output"],
#     {
#         "input": x,
#         "input_lengths": x_lengths,
#         "scales": scales,
#     },
# )
#
# print("done")
#
# input = {
#         "input": x,
#         "input_lengths": x_lengths,
#         "scales": scales,
#     }
#
# onnx_sess.run(output_names=["output"], input_feed=input)
#
# print("done")
#
# subjects = ["I", "You", "Bob", "Alice", "The cat", "The robot"]
# verbs = ["like", "hate", "see", "touch", "admire", "love"]
# objects = ["apples", "the moon", "the rain", "a beautiful painting", "the idea of existence", "the sound of the ocean"]
#
# sentences = []
#
# for i in range(50):
#     subject = random.choice(subjects)
#     verb = random.choice(verbs)
#     object = random.choice(objects)
#     sentence = f"{subject} {verb} {object}."
#     sentences.append(sentence)
#
# x = [numpy.asarray(vits.tokenizer.text_to_ids(sentence, language="en"), dtype=numpy.int64)[None, :] for sentence in sentences]
# x_lengths = [numpy.array([len(x_i[0])], dtype=numpy.int64) for x_i in x]
# scales = [numpy.array([inference_noise_scale, length_scale, inference_noise_scale_dp], dtype=numpy.float32) for _ in range(50)]
#
# # print({
# #         "input": x[1],
# #         "input_lengths": x_lengths[1],
# #         "scales": scales[1],
# #     },)
#
# audio = onnx_sess.run(
#     ["output"],
#     {
#         "input": x[1],
#         "input_lengths": x_lengths[1],
#         "scales": scales[1],
#     },
# )
#
# print("done")
#
# class DataReader(CalibrationDataReader):
#     def __init__(self, x, x_lengths, scales):
#         self.data1 = x
#         self.data2 = x_lengths
#         self.data3 = scales
#         self.pos = -1
#
#     def get_next(self):
#         if self.pos >= len(self.data1) - 1:
#             return None
#         self.pos += 1
#         return {'input': self.data1[self.pos],
#                 'input_lengths': self.data2[self.pos],
#                 'scales': self.data3[self.pos]}
#     def rewind(self):
#         self.pos = -1
#
#
#
# quantize_name = model_name + ".qdq.onnx"
#
# quantize_static(model_name,
#                 quantize_name,
#                 calibration_data_reader=DataReader(x, x_lengths, scales),
#                 quant_format=QuantFormat.QDQ)
#
#




