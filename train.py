import onnx
from onnxruntime.training import artifacts
import onnxruntime.training.api as orttraining
import sys
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data import imagenet
from collections import namedtuple
import multiprocessing
from onnxruntime import InferenceSession
from onnxruntime.capi import _pybind_state as C
import numpy as np
import evaluate
import io
import onnxruntime.training.onnxblock as onnxblock
from onnxruntime.training.api import CheckpointState, Module, Optimizer
from onnxruntime import InferenceSession
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import pdb


# Util function to convert logits to predictions.
def get_pred(logits):
    return np.argmax(logits, axis=1)

# Training Loop :
def train(epoch):
    model.train()
    losses = []
    for _, (data, target) in enumerate(train_loader):
        forward_inputs = [data.reshape(len(data),3,224,224).numpy().astype(np.float32),target.numpy().astype(np.int64)]
        for i in forward_inputs:
            print(i.shape)
        # breakpoint()
        train_loss, _ = model(*forward_inputs)
        optimizer.step()
        model.lazy_reset_grad()
        losses.append(train_loss)

    print(f'Epoch: {epoch+1},Train Loss: {sum(losses)/len(losses):.4f}')

# Test Loop :
def test(epoch):
    model.eval()
    losses = []
    metric = evaluate.load('accuracy')

    for _, (data, target) in enumerate(train_loader):
        forward_inputs = [data.reshape(len(data),3,224,224).numpy().astype(np.float32),target.numpy().astype(np.int64)]
        test_loss, logits = model(*forward_inputs)
        metric.add_batch(references=target, predictions=get_pred(logits))
        losses.append(test_loss)

    metrics = metric.compute()
    print(f'Epoch: {epoch+1}, Test Loss: {sum(losses)/len(losses):.4f}, Accuracy : {metrics["accuracy"]:.2f}')
    

# pdb.set_trace()
print("STARTING")

# Define image transforms
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transform_test = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize
])

# # Load and process input
# train_data = gluon.data.DataLoader(
#    imagenet.classification.ImageNet("/home/TrainingDataset", train=True).transform_first(transform_test),
#    batch_size=batch_size, shuffle=False, num_workers=num_workers)
print("RUNNING")
imagenet_data = datasets.ImageNet('/home/Imagenet', split="train", transform=transform_test)
train_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=4)

print("DATASET LOADED")
# Instantiate the training session by defining the checkpoint state, module, and optimizer
# The checkpoint state contains the state of the model parameters at any given time.
checkpoint_state = orttraining.CheckpointState.load_checkpoint(
    "checkpoint")

model = orttraining.Module(
    "training_model.onnx",
    checkpoint_state,
    "eval_model.onnx",
    "cuda"
)

optimizer = orttraining.Optimizer(
    "optimizer_model.onnx", model
)

print("TRAINING")
for epoch in range(5):
    train(epoch)
    test(epoch)

# ort training api - export the model for so that it can be used for inferencing
model.export_model_for_inferencing("inference.onnx", ["output"])


# num_batches = int((1300*1000)/batch_size)
# num_epochs = 1
# num_samples_per_class = 1300
# # Training loop
# for epoch in range(num_epochs):
#    model.train()
#    loss = 0
#    print('[0 / %d] batches done'%(num_batches))
#    # Loop over batches
#    for i, batch in enumerate(train_data):
#       # Load batch
#       datas = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
#       labels = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
#       print(labels, labels[0], labels[0][0])
#       labels = np.array(labels[0], dtype=np.int64)
#       print(labels)
#       print(datas[0].dtype, datas[0].shape, labels[0].dtype, labels.shape)

#       # ort training api - training model execution outputs the training loss and the parameter gradients
#       loss += model(datas[0].asnumpy().astype(np.float32), labels)
#       # ort training api - update the model parameters by taking a step in the direction of the gradients
#       optimizer.step()
#       # ort training api - reset the gradients to zero so that new gradients can be computed in the next run
#       model.lazy_reset_grad()
#       if (i+1)%100==0:
#             print('[%d / %d] batches done'%(i+1,num_batches))

#    print(f"Epoch {epoch+1} Loss {loss/num_samples_per_class}")


