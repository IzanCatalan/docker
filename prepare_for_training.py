import onnx
from onnxruntime.training import artifacts
import sys
import torch
import torchvision
# Load the onnx model.
# onnx_model = onnx.load(sys.argv[1])
# onnx_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)

# model_name = "resnet50"
# torch.onnx.export(onnx_model, torch.randn(1, 3, 224, 224),
#                   f"training_artifacts/{model_name}.onnx",
#                   input_names=["input"], output_names=["output"],
#                   dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})

onnx_model = onnx.load(sys.argv[1])

requires_grad = ["resnetv17_conv0_weight"]
frozen_params = [
   param.name
   for param in onnx_model.graph.initializer
   if param.name not in requires_grad
]

# onnx_model = onnx.load(f"training_artifacts/resnet50.onnx")
# requires_grad = ["onnx::Conv_497"]
# frozen_params = [
#    param.name
#    for param in onnx_model.graph.initializer
#    if param.name not in requires_grad
# ]

# Generate the training artifacts.
artifacts.generate_artifacts(
   onnx_model,
   requires_grad=requires_grad,
   frozen_params=frozen_params,
   loss=artifacts.LossType.CrossEntropyLoss,
   optimizer=artifacts.OptimType.AdamW,
   artifact_directory=sys.argv[2]
)