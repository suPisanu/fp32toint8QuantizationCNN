import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import numpy as np
import torch.nn.quantized as nnq

from m_copy import NeuralNetwork
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import ToTensor
 
def to_c_array(array, indent = 0) :
    if isinstance(array[0], (list, np.ndarray)) : 
        return " " * indent + "{\n" + ",\n".join(to_c_array(a, indent + 4) for a in array) + "\n" + " " * indent + "}"
    else : 
        return " " * indent + "{" + ", ".join(str(int(x)) for x in array) + "}"

torch.set_printoptions(profile="full")

# Define and prepare model again
model = NeuralNetwork()
model.eval()

model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
torch.ao.quantization.prepare(model, inplace=True)
model_quantized = torch.ao.quantization.convert(model)

# Load weights
model_quantized.load_state_dict(torch.load("model_quantized_state_dict.pth"))

#select target layers
targeted_layer = {"conv1", "conv2"}

#print weight of each selected layers
with open ('weight_int8_export.c', 'w') as f :
    #f.write("#include <stdint.h>\n") #Generate Header File
  
    for name, module in model_quantized.named_modules():
        if name in targeted_layer and isinstance(module, (nnq.Conv2d, nnq.Linear)):
            weight = module.weight()
            shape = weight.shape
            weight_int8 = weight.int_repr().numpy()

            dim = "][".join(str(d) for d in shape)
            f.write(f"int {name}_weight[{dim}] = \n")
            f.write(to_c_array(weight_int8))
            f.write(";\n")

            ##THIS CODE BELOW HERE IS TO PUT EVERYTHING INTO TEXT FILE...##

            #f.write("Raw quantized weights (int_repr): \n")
            #f.write(f"{weight.int_repr()} \n")  # Raw int8 values

            #f.write(f"QScheme: {weight.qscheme()} \n")

            #if weight.qscheme() == torch.per_tensor_affine:
            #    f.write(f"Scale: {weight.q_scale()} \n")
            #    f.write(f"Zero Point: {weight.q_zero_point()} \n")
            #elif weight.qscheme() == torch.per_channel_affine:
            #    f.write(f"Scales: {weight.q_per_channel_scales()} \n")
            #    f.write(f"Zero Points: {weight.q_per_channel_zero_points()} \n")
            #    f.write(f"Axis: {weight.q_per_channel_axis()} \n")
            #else:
            #    f.write("Unknown or unsupported qscheme. \n")
            #print("\n")