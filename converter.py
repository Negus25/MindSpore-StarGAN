import math
import numpy as np
import torch

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import set_seed, Tensor
from mindspore.common import initializer as init

from model import Generator, Discriminator

set_seed(1)
np.random.seed(1)

param = {
'main.0.weight': 'model.0.weight',
'main.1.gamma': 'model.1.weight',
'main.1.beta': 'model.1.bias',
'main.3.weight': 'model.3.weight',
'main.4.gamma': 'model.4.weight',
'main.4.beta': 'model.4.bias',
'main.6.weight': 'model.6.weight',
'main.7.gamma': 'model.7.weight',
'main.7.beta': 'model.7.bias',
'main.9.main.0.weight': 'model.9.conv_block.0.weight',
'main.9.main.1.gamma': 'model.9.conv_block.1.weight',
'main.9.main.1.beta': 'model.9.conv_block.1.bias',
'main.9.main.3.weight': 'model.9.conv_block.3.weight',
'main.9.main.4.gamma': 'model.9.conv_block.4.weight',
'main.9.main.4.beta': 'model.9.conv_block.4.bias',
'main.10.main.0.weight': 'model.10.conv_block.0.weight',
'main.10.main.1.gamma': 'model.10.conv_block.1.weight',
'main.10.main.1.beta': 'model.10.conv_block.1.bias',
'main.10.main.3.weight': 'model.10.conv_block.3.weight',
'main.10.main.4.gamma': 'model.10.conv_block.4.weight',
'main.10.main.4.beta': 'model.10.conv_block.4.bias',
'main.11.main.0.weight': 'model.11.conv_block.0.weight',
'main.11.main.1.gamma': 'model.11.conv_block.1.weight',
'main.11.main.1.beta': 'model.11.conv_block.1.bias',
'main.11.main.3.weight': 'model.11.conv_block.3.weight',
'main.11.main.4.gamma': 'model.11.conv_block.4.weight',
'main.11.main.4.beta': 'model.11.conv_block.4.bias',
'main.12.main.0.weight': 'model.12.conv_block.0.weight',
'main.12.main.1.gamma': 'model.12.conv_block.1.weight',
'main.12.main.1.beta': 'model.12.conv_block.1.bias',
'main.12.main.3.weight': 'model.12.conv_block.3.weight',
'main.12.main.4.gamma': 'model.12.conv_block.4.weight',
'main.12.main.4.beta': 'model.12.conv_block.4.bias',
'main.13.main.0.weight': 'model.13.conv_block.0.weight',
'main.13.main.1.gamma': 'model.13.conv_block.1.weight',
'main.13.main.1.beta': 'model.13.conv_block.1.bias',
'main.13.main.3.weight': 'model.13.conv_block.3.weight',
'main.13.main.4.gamma': 'model.13.conv_block.4.weight',
'main.13.main.4.beta': 'model.13.conv_block.4.bias',
'main.14.main.0.weight': 'model.14.conv_block.0.weight',
'main.14.main.1.gamma': 'model.14.conv_block.1.weight',
'main.14.main.1.beta': 'model.14.conv_block.1.bias',
'main.14.main.3.weight': 'model.14.conv_block.3.weight',
'main.14.main.4.gamma': 'model.14.conv_block.4.weight',
'main.14.main.4.beta': 'model.14.conv_block.4.bias',
'main.15.weight': 'model.15.weight',
'main.16.gamma': 'model.16.weight',
'main.16.beta': 'model.16.bias',
'main.18.weight': 'model.18.weight',
'main.19.gamma': 'model.19.weight',
'main.19.beta': 'model.19.bias',
'main.21.weight': 'model.21.weight',
}
# Print the parameter names and shapes of all parameters in the PyTorch parameter file and return the parameter dictionary.
def pytorch_params(pth_file):
    par_dict = torch.load(pth_file, map_location='cpu')
    pt_params = {}
    for name in par_dict:
        parameter = par_dict[name]
        print(name, parameter.numpy().shape)
        pt_params[name] = parameter.numpy()
    return pt_params

# Print the names and shapes of all parameters in the MindSpore cell and return the parameter dictionary.
def mindspore_params(network):
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        print(name, value.shape)
        ms_params[name] = value
    return ms_params

def param_convert(ms_params, pt_params, ckpt_path):
    # Parameter name mapping dictionary
    new_params_list = []
    for ms_param in ms_params.keys():
  
            # If the corresponding parameter is found and the shape is the same, add the parameter to the parameter list.
            if param[ms_param] in pt_params and pt_params[param[ms_param]].shape == ms_params[ms_param].shape:
                ms_value = pt_params[param[ms_param]]
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
            else:
                print(ms_param, "not match in pt_params")
    ms.save_checkpoint(new_params_list, ckpt_path)
    

g_conv_dim=64
c_dim=5
g_repeat_num=6
d_repeat_num=6
d_conv_dim=64
image_size=128
G = Generator(g_conv_dim, c_dim, g_repeat_num)
D = Discriminator(image_size, d_conv_dim, c_dim, d_repeat_num)

pth_path = "generator_49.pth"
pt_param = pytorch_params(pth_path)
print("="*20)
ms_param = mindspore_params(G)


ckpt_path = "torch_model.ckpt"
param_convert(ms_param, pt_param, ckpt_path)