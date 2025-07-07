import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import numpy as np
import torch.nn.quantized as nnq

from trainer import NeuralNetwork
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
model_quantized.load_state_dict(torch.load("C:\\WORK\\FPGA\\Litterally_Work\\fp32toint8QuantizationCNN_new\\model\\model_quantized_state_dict_11.pth"))

#select target layers
targeted_layer = ["conv1", "conv2"]

header_guard = "__WEIGHT_INT8_EXPORT_H__"

layer_mapping = {old : f'kernel_layer_{i}' for i, old in enumerate(targeted_layer)}
macro_lines = []
weight_lines = []

c = 0
for name, module in model_quantized.named_modules():
    if name in targeted_layer and isinstance(module, (nnq.Conv2d, nnq.Linear)):
        weight = module.weight()
        shape = weight.shape
        
        weight_int8 = weight.int_repr().numpy()
        #Transpose from [32, 1, 3, 3] to [1, 32, 3, 3]
        weight_int8 = np.transpose(weight_int8, (1, 0, 2, 3))
        weight_int8_shape = weight_int8.shape

        mapped_name = layer_mapping[name]
        #macro_lines.append(f"#define {mapped_name.upper()}_INPUT_CH_{c} {weight_int8_shape[0]}\n")
        #macro_lines.append(f"#define {mapped_name.upper()}_FILTER_CH_{c} {weight_int8_shape[1]}\n")
        #macro_lines.append(f"#define {mapped_name.upper()}_WIDTH_{c} {weight_int8_shape[2]}\n")
        #macro_lines.append(f"#define {mapped_name.upper()}_HEIGHT_{c} {weight_int8_shape[3]}\n\n")

        macro_lines.append(f"#define CONV_INPUT_CH_{c} {weight_int8_shape[0]}\n")
        macro_lines.append(f"#define CONV_FILTER_CH_{c} {weight_int8_shape[1]}\n")
        macro_lines.append(f"#define CONV_PE_REAL_{c} {weight_int8_shape[2]}\n\n")   

        dim = "][".join(str(d) for d in weight_int8_shape)
        weight_c_arr = (to_c_array(weight_int8))
        weight_lines.append(f"int {mapped_name}_weight[{dim}] = \n{weight_c_arr};\n\n")
        c += 1

with open ('weight_int8_export.h', 'w') as f :
    f.write(f"#ifndef {header_guard}\n")
    f.write(f"#define {header_guard}\n\n")

    f.writelines(macro_lines)
    f.writelines(weight_lines)

    f.write(f"#endif // {header_guard}\n")

    ##THIS CODE BELOW HERE IS TO PUT EVERYTHING INTO TEXT FILE...#
    #f.write("Raw quantized weights (int_repr): \n")
    #f.write(f"{weight.int_repr()} \n")  # Raw int8 value
    #f.write(f"QScheme: {weight.qscheme()} \n"
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
        