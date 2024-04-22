import numpy as np
import cv2
import os
from ctypes import *
import time

path = "../data/ssdlite_mobiledet_coco_int8_vim4.adla"
if path is None:
    raise ValueError("No model path provided")

adla_path = os.path.splitext(path)[0] + ".adla"
model_path = adla_path.encode('utf-8')
if not path.endswith(".adla"):
    raise Exception(f"unknown model format {path}")

try:
    so_file = "./libadla_interface.so"
    interface = cdll.LoadLibrary(so_file)
except OSError as e:
    raise print(
        "ERROR: failed to load library. %s",
        e,
    )

labels = {}
with open("../data/coco_labels.txt") as f:
    for idx, val in enumerate(f):
        labels[idx] = val.strip()

#void* init_network_file(const char *mpath)
interface.init_network_file.argtypes = [c_char_p]
interface.init_network_file.restype = c_void_p

try:
    context = interface.init_network_file(model_path)
except OSError as e:
    raise logger.error("ERROR: failed to initialize NPU with model %s: %s", model_path, e)

# read the image
MODEL_WIDTH=320
MODEL_HEIGHT=320

img = cv2.imread("../data/horses.jpg")

img_model = np.zeros((MODEL_WIDTH, MODEL_HEIGHT, 1), dtype = "uint8")
img_resized = cv2.resize(img, (MODEL_WIDTH, MODEL_HEIGHT), img_model)
temp_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

image = temp_img.ctypes.data_as(POINTER(c_ubyte))

interface.set_input.argtypes = [c_void_p, POINTER(c_ubyte), c_int]
interface.set_input.restype = c_int

ret = interface.set_input(context, image, temp_img.size)

print(f"set input returns {ret}")

#int run_network(void *qcontext)

class detBox(Structure):
    _fields_ = [("ymin", c_float),
                ("xmin", c_float),
                ("ymax", c_float),
                ("xmax", c_float),
                ("score", c_float),
                ("objectClass", c_float)]

interface.run_network.argtypes = [c_void_p, POINTER(c_uint32), POINTER(detBox)]
interface.run_network.restype = c_int

box = np.zeros((6, 230), dtype=[('ymin', np.float32), ('xmin', np.float32), ('ymax', np.float32), ('xmax', np.float32),
                                ('score', np.float32), ('objectClass', np.float32)])
count = pointer(c_uint32(0))
start_time = time.time()
ret = interface.run_network(context, count, box.ctypes.data_as(POINTER(detBox)))

print(f"run network returns {ret}")

stop_time = time.time()
duration = stop_time - start_time
print(f"run network returns {ret} after {duration} seconds")

for i in range(count[0]):
    print(f"object number {i} class {labels[int(box['objectClass'][0][i])]}")
    print(f"object number {i} score {float(box['score'][0][i])}")
    print(f"object number {i} ymin {float(box['ymin'][0][i])}")
    print(f"object number {i} xmin {float(box['xmin'][0][i])}")
    print(f"object number {i} ymax {float(box['ymax'][0][i])}")
    print(f"object number {i} xmax {float(box['xmax'][0][i])}")

#int destroy_network(void *qcontext)

interface.destroy_network.argtypes = [c_void_p]
interface.destroy_network.restype = c_int

ret = interface.destroy_network(context)

print(f"destroy network returns {ret}")
