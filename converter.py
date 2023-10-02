import onnx
import sys 
from onnx import version_converter

onnx_model = onnx.load(sys.argv[1])
# converted_model = version_converter.convert_version(onnx_model, 14)
# onnx.save(converted_model, sys.argv[2])

opset_version = onnx_model.opset_import[0].version if len(onnx_model.opset_import) > 0 else None

print(opset_version)

# Set the target opset version to 17
# target_opset_version = 17

# # Iterate through the model's graph and update the opset version
# for node in onnx_model.graph.node:
#     node.opset_import[0].version = target_opset_version

# # Save the modified model with opset version 17
# onnx.save(onnx_model, "model_opset17.onnx")