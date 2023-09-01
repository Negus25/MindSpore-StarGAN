import os
import math
import numpy as np
from PIL import Image

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore import dtype as mstype
import mindspore.ops as ops
from PIL import Image
from mindspore.train.serialization import load_param_into_net
from mindspore import load_checkpoint
from mindspore import Tensor, context

from utils import create_labels, denorm, init_weights
from model import Generator, Discriminator, GeneratorLoss, DiscriminatorLoss, ClassificationLoss, WGANGPGradientPenalty
from dataset import dataloader, DistributedSampler

data_dir = "/home/negus/CelebA/Img/img_align_celeba/"  # Root directory of the dataset
batch_size = 16 # Batch size
image_size = 32  # Image size of training data
workers = 4  # Number of parallel workers
num_classes = 10  # Number of classes
img_height  = 128
img_width =128


data_path = '/home/negus/CelebA/Img/img_align_celeba/'
attr_path = '/home/negus/CelebA/Img/img_align_celeba/list_attr_celeba.txt'
local_train_url = './models/'
batch_size=16
selected_attrs=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
device_num=1
dataset='CelebA'
mode='train'
local_data_url = './king/data'
local_train_url = '/king/ckpt'
result_dir = './results'
device_id=0

g_conv_dim=64
c_dim=5
g_repeat_num=6
d_repeat_num=6
d_conv_dim=64
image_size=128

local_data_url = os.path.join(local_data_url, str(device_id))
local_train_url = os.path.join(local_train_url, str(device_id))


device_target='GPU'

context.set_context(mode=context.GRAPH_MODE, device_target=device_target,
                            device_id=device_id, save_graphs=False)


G = Generator(g_conv_dim, c_dim, g_repeat_num)
#D  = Discriminator(image_size, d_conv_dim, c_dim, d_repeat_num)
init_weights(G, 'KaimingUniform', math.sqrt(5))
#init_weights(D, 'KaimingUniform', math.sqrt(5))

G_path = 'Generator-0_5000.ckpt'
    # D_path = os.path.join(config.model_save_dir, f"Net_D_%d.ckpt" % config.resume_iters)
param_G = load_checkpoint(G_path, G)
load_param_into_net(G, param_G)
G.set_train(False)


dataset, length = dataloader(img_path=data_path,
                                     attr_path=attr_path,
                                     batch_size=batch_size,
                                     selected_attr=selected_attrs,
                                     device_num=device_num,
                                     dataset=dataset,
                                     mode='val',
                                     shuffle=True)

op = ops.Concat(axis=3)
ds = dataset.create_dict_iterator()
print(length)
print('Start Evaluating!')
for i, data in enumerate(ds):
        result_list = ()
        img_real = denorm(data['image'].asnumpy())
        x_real = Tensor(data['image'], mstype.float32)
        result_list += (x_real,)
        c_trg_list = create_labels(data['attr'].asnumpy(), selected_attrs=selected_attrs)
        c_trg_list = Tensor(c_trg_list, mstype.float32)
        x_fake_list = []

        for c_trg in c_trg_list:

            x_fake = G(x_real, c_trg)
            x = Tensor(x_fake.asnumpy().copy())

            result_list += (x,)

        x_fake_list = op(result_list)

        result = denorm(x_fake_list.asnumpy())
        result = np.reshape(result, (-1, 768, 3))

        im = Image.fromarray(np.uint8(result))
        im.save(result_dir + '/test_{}.jpg'.format(i))
        print('Successful save image in ' + result_dir + '/test_{}.jpg'.format(i))