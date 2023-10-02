from operator import eq
from statistics import mode
import numpy as np
import onnx
import os
import glob
import onnxruntime as backend
import onnx.numpy_helper as oh
from PIL import Image
from onnx import numpy_helper
import matplotlib.pyplot as plt
import json
from datetime import datetime

from typing import Iterable

import random
import struct
import ctypes
from struct import *

from struct import *

import io
import sys
import time


import onnx
import onnxruntime
import ssl
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data import imagenet
from collections import namedtuple
import multiprocessing
with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

def writeFiles(indexes, preds, order, check1, check5):
    for i,f in enumerate(files):
        # file = open("prediction_%i.txt" %i,'a')
        if i < 1:
            # f.write("%s class=%s ; probability=%f\n" %(order, labels[indexes[i]],preds[indexes[i]]))
            f.write(f"{order} probability= {preds[indexes[i]]} top1= {check1} top5= {check5[0]} top5pos= {check5[1]} class= {labels[indexes[i]]}\n")
        else:
            # f.write("%s class=%s ; probability=%f\n" %(order, labels[indexes[i]],preds[indexes[i]]))
            f.write(f"{order} probability= {preds[indexes[i]]} class= {labels[indexes[i]]}\n")
    return

def openFiles():
    for f in files:
        f.write("BEGIN\n")
    return


def closeFiles():
    for f in files:
        f.write("END\n")
        f.close()
    return

def bitflipDouble(x,pos):
    fs = pack('d',x)
    bval = list(unpack('BBBBBBBB',fs))
    [q,r] = divmod(pos,8)
    bval[q] ^= 1 << r
    fs = pack('BBBBBBBB', *bval)
    fnew=unpack('d',fs)
    return fnew[0]

def bitflipFloat(x,pos):
    fs = pack('f',x)
    bval = list(unpack('BBBB',fs))
    [q,r] = divmod(pos,8)
    bval[q] ^= 1 << r
    fs = pack('BBBB', *bval)
    fnew=unpack('f',fs)
    return fnew[0]


def insert_fail(initializer):

    name = initializer.name
    a = onnx.numpy_helper.to_array(initializer).copy()
    # print("Tensor information:")
    print(f"Tensor Name: {name}")
    print(f"Data Type: {a.dtype}")
    print(f"Shape: {initializer.dims}")
    print(f"Size: {a.size}")
    # bit flip of a random element of a random initialize
    if initializer.data_type == onnx.TensorProto.DataType.DOUBLE or initializer.data_type == onnx.TensorProto.DataType.UINT64 or initializer.data_type == onnx.TensorProto.DataType.INT64:
        end = 64
    elif initializer.data_type == onnx.TensorProto.DataType.UINT32 or initializer.data_type == onnx.TensorProto.DataType.INT32 or initializer.data_type == onnx.TensorProto.DataType.FLOAT:
        end = 32
    else:
        end = 8

    random.seed(datetime.now())
    random_bit = random.randint(0, end - 1)
    random.seed(datetime.now())
    random_elem = random.randint(0,a.size - 1)
    number = a.flat[random_elem]
    print(f"Number before NB= {number} :")
    if initializer.data_type == onnx.TensorProto.DataType.FLOAT:    
        print(bin(ctypes.c_uint32.from_buffer(ctypes.c_float(number)).value))
        number = bitflipFloat(number, random_bit)
        print(f"Number float after NFA= {number} :")
        print(bin(ctypes.c_uint32.from_buffer(ctypes.c_float(number)).value))
    elif initializer.data_type == onnx.TensorProto.DataType.DOUBLE:
        print(bin(ctypes.c_uint32.from_buffer(ctypes.c_float(number)).value))
        number = bitflipDouble(number, random_bit)
        print(f"Number double after NDA= {number} :")
        print(bin(ctypes.c_uint64.from_buffer(ctypes.c_float(number)).value))
    else:
        # integers or uints types
        print(bin(number))
        number = number ^ (1 << random_bit)
        print(f"Number integer after {number} :")
        print(bin(number))


    a.flat[random_elem] = number
    u = onnx.numpy_helper.from_array(a, name)
    value.CopyFrom(u)
    print(f"num elems NE= {a.size}")
    print(f"random bit RB= {random_bit}")
    print(f"random elem RE= {random_elem}")
    return

def evaluate(model_path, data_dir, order):

    print(f"--------------------------------------VALIDATE {order} {model_path} --------------------------------------")
    top1 = None
    top5 = None
    acc_top1 = None
    acc_top5 = None
    # Determine and set context
    if len(mx.test_utils.list_gpus())==0:
        ctx = [mx.cpu()]
    else:
        ctx = [mx.gpu()]

    # batch size (set to 1 for cpu)
    batch_size = 1

    # number of preprocessing workers
    num_workers = multiprocessing.cpu_count()

    # Define evaluation metrics
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)

    # Define image transforms
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    # Load and process input
    val_data = gluon.data.DataLoader(
        imagenet.classification.ImageNet(data_dir, train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)

    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    ort_session_cpu = onnxruntime.InferenceSession(onnx_model.SerializeToString(),providers=['CUDAExecutionProvider'])


    # Compute evaluations
    print("----Running ONNXRuntime----")
    num_batches = int(50000/batch_size)
    print('[0 / %d] batches done'%(num_batches))
    ort_start = time.time()
    ort_latency = []


    # Loop over batches
    for i, batch in enumerate(val_data):
        # Load batch
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        # Perform forward pass
        ort_inputs_cpu = {ort_session_cpu.get_inputs()[0].name: data[0].asnumpy()}
        outputs=ort_session_cpu.run(None, ort_inputs_cpu)
        # Update accuracy metrics
        list = [[]]
        # preds = mx.nd.array(outputs[0][0])
        list[0] = mx.nd.array(outputs[0])
        #perform stats with predictions and write them on files
        preds = mx.ndarray.softmax(list[0][0]).asnumpy()
        preds = np.squeeze(preds)
        a = np.argsort(preds)[::-1]
        # print(mx.nd.shape_array(list[0][0]))
        check1 = acc_top1.update(label, list)
        check5 = acc_top5.update(label, list)
        ort_latency.append(time.time() - ort_start)
        writeFiles(a, preds, order, check1, check5)
        preds.sort()
        predictions.append(preds)
        if (i+1)%num_batches==0:
            print('[%d / %d] batches done'%(i+1,num_batches))


    ort_avg_time = (sum(ort_latency) * 1000 / len(ort_latency))
    ort_fps = batch_size/((ort_avg_time)/1000)

    # Print results
    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    print(f"Batch size {batch_size}")
    print("Top-1: {}".format(top1))
    print("Top-5: {}".format(top5))
    print("Average onnxruntime Inference time= {} ms".format(ort_avg_time, '.2f'))
    print("Average onnxruntime fps= {} ".format(ort_fps, '.2f'))
    top1 = None
    top5 = None
    acc_top1 = None
    acc_top5 = None
    return



# ---------------------------------------------------MAIN--------------------------------------------------------------------------
print("\nBEGIN")

# path to imagenet dataset folder
data = '/mnt/beegfs/gap/izcagal/models/vision/classification/ValidateDataset'
predictions = []
files = [open("prediction_%i.txt" %i,'a') for i in range(int(sys.argv[3]))]
openFiles()
# path to ONNX model file
old_model = sys.argv[1]
new_model = sys.argv[2]
# ------------------------------------------------CHECK ACCURACY--------------------------------------------------------------------
# Evaluate del modelo sin fallo
evaluate(old_model, data, "B")

# -------------------------------------------------INSERT FAIL----------------------------------------------------------------------
# Insertamos 1 bit flip aleatorio en un elemento aleatorio de 1 tensor aleatorio de entre los inputs y lo guardamos en un nuevo onnx
print("--------------------------------------INSERT FAIL--------------------------------------")
resnet = onnx.load(old_model)
print("LOAD")
onnx.checker.check_model(resnet)
nodes = resnet.graph.node
initializers = resnet.graph.initializer
random.seed(datetime.now())
n = random.choice(nodes)

while "conv" not in n.name:
    n = random.choice(nodes)
    
print(f"Module= {n.name}")
weight_name = n.input[1] 
value = None
for i in initializers:
    if i.name == weight_name:
        value = i
        break

insert_fail(value)
print("INSERTED")
onnx.checker.check_model(resnet)
onnx.save(resnet, new_model)
old_predictions = predictions.copy()
predictions.clear()
print (predictions)

# -------------------------------------------------CHECK NEW ACCURACY----------------------------------------------------------------
# evaluamos el nuevo modelo con fallo convnets_modified.onnx
evaluate(new_model, data, "F")
closeFiles()
eque = 0
diff = 0
diffs = []
rtol = 1e-05
atol = 1e-08

for i in range (len(predictions)):
    if np.allclose(predictions[i],old_predictions[i], rtol, atol, equal_nan = True):
        eque = eque+1
    else:
        diff = diff+1
        diffs.append(i)

print(f"Predictions equals: {eque}, Predictions not equals: {diff} with relative tolerance of {rtol} and absolute tolerance of {atol}\n")

# for e in diffs:
#     print(f"\n\nPrediction Image {e}\n")
#     try:
#         np.testing.assert_allclose(predictions[e],old_predictions[e], rtol, atol, equal_nan=True)
#     except AssertionError as p:
#         print(p)


print("END")