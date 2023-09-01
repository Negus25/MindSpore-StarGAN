# +
"""Train the model."""
from time import time
import os
import argparse
import ast
import numpy as np
import math

import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore import Tensor, context
import mindspore.ops as P
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, _InternalCallbackParam, RunContext
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore import dtype as mstype
from PIL import Image
import mindspore.ops.functional as F



from dataset import dataloader, DistributedSampler
from utils import init_weights
from model import Generator, Discriminator, GeneratorLoss, DiscriminatorLoss, ClassificationLoss, WGANGPGradientPenalty
from reporter import Reporter   




class GeneratorWithLossCell(nn.Cell):
   
    def __init__(self, network):
        super(GeneratorWithLossCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, x_real, c_org, c_trg):
        _, G_Loss, _, _, _, = self.network(x_real, c_org, c_trg)
        return G_Loss


class DiscriminatorWithLossCell(nn.Cell):

    def __init__(self, network):
        super(DiscriminatorWithLossCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, x_real, c_org, c_trg):
        D_Loss, _, _, _, _ = self.network(x_real, c_org, c_trg)
        return D_Loss
    

class TrainOneStepCellGen(nn.Cell):
    def __init__(self, G, optimizer, sens=1.0):
        super(TrainOneStepCellGen, self).__init__()
        self.optimizer = optimizer
        self.G = G
        self.G.set_grad()
        self.G.set_train()
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = optimizer.parameters
        self.network = GeneratorWithLossCell(G)
        self.network.add_flags(defer_inline=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity

    def construct(self, img_real, c_org, c_trg):
        weights = self.weights
        fake_image, loss, G_fake_loss, G_fake_cls_loss, G_rec_loss = self.G(img_real, c_org, c_trg)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(img_real, c_org, c_trg, sens)
        grads = self.grad_reducer(grads)

        return F.depend(loss, self.optimizer(grads)), fake_image, loss, G_fake_loss, G_fake_cls_loss, G_rec_loss



class TrainOneStepCellDis(nn.Cell):
    def __init__(self, D, optimizer, sens=1.0):
        super(TrainOneStepCellDis, self).__init__()
        self.optimizer = optimizer
        self.D = D
        self.D.set_grad()
        self.D.set_train()
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = optimizer.parameters
        self.network = DiscriminatorWithLossCell(D)
        self.network.add_flags(defer_inline=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity
        

    def construct(self, img_real, c_org, c_trg):
        weights = self.weights

        loss, D_real_loss, D_fake_loss, D_real_cls_loss, D_gp_loss = self.D(img_real, c_org, c_trg)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(img_real, c_org, c_trg, sens)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)

        return F.depend(loss, self.optimizer(grads)), loss, D_real_loss, D_fake_loss, D_real_cls_loss, D_gp_loss



data_dir = "/home/negus/CelebA/Img/img_align_celeba/"  # Root directory of the dataset

image_size = 32  # Image size of training data
workers = 4  # Number of parallel workers
num_classes = 10  # Number of classes
img_height  = 128
img_width =128

# +
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
device_id=0
g_conv_dim=64
c_dim=5
g_repeat_num=6
d_repeat_num=6
d_conv_dim=64
image_size=128

# Define network with loss
lambda_rec = 10
lambda_cls = 1
lambda_gp = 10

beta1=0.5
beta2=0.999
g_lr=0.0001
d_lr=0.0001

batch_size = 16 # Batch size
star_iter = 0
iter_sum = 200000
num_iters=200000
model_save_step=5000
batch_size=4
n_critic=5
log_step=10
epochs=59


local_data_url = os.path.join(local_data_url, str(device_id))
local_train_url = os.path.join(local_train_url, str(device_id))


device_target='GPU'
context.set_context(mode=context.GRAPH_MODE, device_target=device_target,
                            device_id=device_id, save_graphs=False)

dataset, length = dataloader(img_path=data_path,
                                     attr_path=attr_path,
                                     batch_size=batch_size,
                                     selected_attr=selected_attrs,
                                     device_num=device_num,
                                     dataset=dataset,
                                     mode=mode,
                                     shuffle=True)

print(length)
dataset_iter = dataset.create_dict_iterator()



generator = Generator(g_conv_dim, c_dim, g_repeat_num)
discriminator  = Discriminator(image_size, d_conv_dim, c_dim, d_repeat_num)
init_weights(generator, 'KaimingUniform', math.sqrt(5))
init_weights(discriminator, 'KaimingUniform', math.sqrt(5))



cls_loss = ClassificationLoss()
wgan_loss = WGANGPGradientPenalty(discriminator)


G_loss_cell = GeneratorLoss(generator, discriminator)
D_loss_cell = DiscriminatorLoss(generator, discriminator)

Optimizer_G = nn.Adam(generator.trainable_params(), learning_rate=g_lr,
                          beta1=beta1, beta2=beta2)
Optimizer_D = nn.Adam(discriminator.trainable_params(), learning_rate=d_lr,
                          beta1=beta1, beta2=beta2)


# Define One step train
G_trainOneStep = TrainOneStepCellGen(G_loss_cell, Optimizer_G)
D_trainOneStep = TrainOneStepCellDis(D_loss_cell, Optimizer_D)


# Train
G_trainOneStep.set_train()
D_trainOneStep.set_train()


print('Start Training')
ckpt_config = CheckpointConfig(save_checkpoint_steps=model_save_step)
ckpt_cb_g = ModelCheckpoint(config=ckpt_config, directory='home/negus/king/ckpt1/', prefix='Generator')
ckpt_cb_d = ModelCheckpoint(config=ckpt_config, directory='home/negus/king/ckpt1/', prefix='Discriminator')


cb_params_g = _InternalCallbackParam()
cb_params_g.train_network = generator
cb_params_g.cur_step_num = 0
cb_params_g.batch_num = 4
cb_params_g.cur_epoch_num = 0

cb_params_d = _InternalCallbackParam()
cb_params_d.train_network = discriminator
cb_params_d.cur_step_num = 0
cb_params_d.batch_num = batch_size
cb_params_d.cur_epoch_num = 0
run_context_g = RunContext(cb_params_g)
run_context_d = RunContext(cb_params_d)
ckpt_cb_g.begin(run_context_g)
ckpt_cb_d.begin(run_context_d)

Reporter = Reporter()

start = time()
for iterator in range(num_iters):
    data = next(dataset_iter)
    x_real = Tensor(data['image'], mstype.float32)
    c_trg = Tensor(data['attr'], mstype.float32)
    c_org = Tensor(data['attr'], mstype.float32)
    np.random.shuffle(c_trg)

    d_out = D_trainOneStep(x_real, c_org, c_trg)

    if (iterator + 1) % n_critic == 0:
        g_out = G_trainOneStep(x_real, c_org, c_trg)

    if (iterator + 1) % log_step == 0:
        Reporter.print_info(start, iterator, g_out, d_out)
        _, _, dict_G, dict_D = Reporter.return_loss_array(g_out, d_out)

    if (iterator + 1) % model_save_step == 0:
        cb_params_d.cur_step_num = iterator + 1
        cb_params_d.batch_num = iterator + 2
        cb_params_g.cur_step_num = iterator + 1
        cb_params_g.batch_num = iterator + 2
        ckpt_cb_g.step_end(run_context_g)
        ckpt_cb_d.step_end(run_context_d)

