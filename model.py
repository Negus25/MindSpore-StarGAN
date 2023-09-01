
# ============================================================================
"""Define Generator and Discriminator for StarGAN"""
import math
import numpy as np

import mindspore.nn as nn

import mindspore.ops as ops
from mindspore import set_seed, Tensor
from mindspore.common import initializer as init
from mindspore import dtype as mstype
import mindspore.ops as P


set_seed(1)
np.random.seed(1)


lambda_rec = 10
lambda_cls = 1
lambda_gp = 10


class ResidualBlock(nn.Cell):
    """Residual Block with group normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.SequentialCell(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=False),
            nn.GroupNorm(num_groups=dim_out, num_channels=dim_out),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=False),
            nn.GroupNorm(num_groups=dim_out, num_channels=dim_out)
        )

    def construct(self, x):
        return x + self.main(x)


class Generator(nn.Cell):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=4, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append((nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1,
                                 padding=3, pad_mode='pad', has_bias=False)))
        layers.append(nn.GroupNorm(num_groups=conv_dim, num_channels=conv_dim))
        layers.append(nn.ReLU())

        # Down-sampling layers.
        curr_dim = conv_dim
        for _ in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2,
                                    padding=1, pad_mode='pad', has_bias=False))
            layers.append(nn.GroupNorm(num_groups=curr_dim*2, num_channels=curr_dim*2))
            layers.append(nn.ReLU())
            curr_dim = curr_dim*2

        # Bottleneck layers.
        for _ in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for _ in range(2):
            layers.append(nn.Conv2dTranspose(curr_dim, int(curr_dim/2), kernel_size=4, stride=2,
                                             padding=1, pad_mode='pad', has_bias=False))
            layers.append(nn.GroupNorm(num_groups=int(curr_dim/2), num_channels=int(curr_dim/2)))
            layers.append(nn.ReLU())
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, pad_mode='pad', has_bias=False))
        layers.append(nn.Tanh())
        self.main = nn.SequentialCell(*layers)

    def construct(self, x, c):
        reshape = P.Reshape()
        c = reshape(c, (c.shape[0], c.shape[1], 1, 1))
        c = P.functional.reshape(c, (c.shape[0], c.shape[1], 1, 1))
        tile = P.Tile()
        c = tile(c, (1, 1, x.shape[2], x.shape[3]))
        op = P.Concat(1)
        x = op((x, c))
        return self.main(x)


class ResidualBlock_2(nn.Cell):
    """Residual Block with instance normalization."""

    def __init__(self, weight, dim_in, dim_out):
        super(ResidualBlock_2, self).__init__()
        self.main = nn.SequentialCell(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1,
                      pad_mode='pad', has_bias=False, weight_init=Tensor(weight[0])),
            nn.GroupNorm(num_groups=dim_out, num_channels=dim_out),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1,
                      pad_mode='pad', has_bias=False, weight_init=Tensor(weight[3])),
            nn.GroupNorm(num_groups=dim_out, num_channels=dim_out)
        )

    def construct(self, x):
        return x + self.main(x)


class Discriminator(nn.Cell):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1, has_bias=True,
                                pad_mode='pad', bias_init=init.Uniform(1 / math.sqrt(3))))
        layers.append(nn.LeakyReLU(alpha=0.01))

        curr_dim = conv_dim
        for _ in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, has_bias=True,
                                    pad_mode='pad', bias_init=init.Uniform(1 / math.sqrt(curr_dim))))
            layers.append(nn.LeakyReLU(alpha=0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.SequentialCell(*layers)
        # Patch GAN输出结果
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, has_bias=False, pad_mode='valid')

    def construct(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        reshape = P.Reshape()
        out_cls = reshape(out_cls, (out_cls.shape[0], out_cls.shape[1]))
        return out_src, out_cls


def generate_tensor(batch_size):
    np_array = np.random.randn(batch_size, 1, 1, 1)
    return Tensor(np_array, mstype.float32)


class ClassificationLoss(nn.Cell):
    """Define classification loss for StarGAN"""
    def __init__(self, dataset='CelebA'):
        super().__init__()
        self.BCELoss = P.BinaryCrossEntropy(reduction='sum')
        self.cross_entropy = P.SoftmaxCrossEntropyWithLogits()
        self.dataset = dataset
        self.bec = nn.BCELoss(reduction='sum')

    def construct(self, pred, label):
        if self.dataset == 'CelebA':
            weight = ops.Ones()(pred.shape, mstype.float32)
            pred_ = P.Sigmoid()(pred)
            x = self.BCELoss(pred_, label, weight) / pred.shape[0]

        else:
            x = self.cross_entropy(pred, label)
        return x


class GradientWithInput(nn.Cell):
    """Get Discriminator Gradient with Input"""
    def __init__(self, discrimator):
        super(GradientWithInput, self).__init__()
        self.reduce_sum = ops.ReduceSum()
        self.discrimator = discrimator

    def construct(self, interpolates):
        decision_interpolate, _ = self.discrimator(interpolates)
        decision_interpolate = self.reduce_sum(decision_interpolate, 0)
        return decision_interpolate


class WGANGPGradientPenalty(nn.Cell):
    """Define WGAN loss for StarGAN"""
    def __init__(self, discrimator):
        super(WGANGPGradientPenalty, self).__init__()
        self.gradient_op = ops.GradOperation()

        self.reduce_sum = ops.ReduceSum()
        self.reduce_sum_keep_dim = ops.ReduceSum(keep_dims=True)
        self.sqrt = ops.Sqrt()
        self.discrimator = discrimator
        self.gradientWithInput = GradientWithInput(discrimator)

    def construct(self, x_real, x_fake):
        """get gradient penalty"""
        batch_size = x_real.shape[0]
        alpha = generate_tensor(batch_size)
        alpha = alpha.expand_as(x_real)
        x_fake = ops.functional.stop_gradient(x_fake)
        x_hat = (alpha * x_real + (1 - alpha) * x_fake)

        gradient = self.gradient_op(self.gradientWithInput)(x_hat)
        gradient_1 = ops.reshape(gradient, (batch_size, -1))
        gradient_1 = self.sqrt(self.reduce_sum(gradient_1*gradient_1, 1))
        gradient_penalty = self.reduce_sum((gradient_1 - 1.0)**2) / x_real.shape[0]
        return gradient_penalty


class GeneratorLoss(nn.Cell):
    """Define total Generator loss"""
    def __init__(self, generator, discriminator):
        super(GeneratorLoss, self).__init__()
        self.net_G = generator
        self.net_D = discriminator
        self.cyc_loss = P.ReduceMean()
        self.rec_loss = nn.L1Loss("mean")
        self.cls_loss = ClassificationLoss()

        self.lambda_rec = lambda_rec
        self.lambda_cls = lambda_cls

    def construct(self, x_real, c_org, c_trg):
        """Get generator loss"""
        # Original to Target
        x_fake = self.net_G(x_real, c_trg)
        fake_src, fake_cls = self.net_D(x_fake)

        G_fake_loss = - self.cyc_loss(fake_src)
        G_fake_cls_loss = self.cls_loss(fake_cls, c_trg)

        # Target to Original
        x_rec = self.net_G(x_fake, c_org)
        G_rec_loss = self.rec_loss(x_real, x_rec)

        g_loss = G_fake_loss + self.lambda_cls * G_fake_cls_loss + self.lambda_rec * G_rec_loss

        return (x_fake, g_loss, G_fake_loss, G_fake_cls_loss, G_rec_loss)


class DiscriminatorLoss(nn.Cell):
    """Define total discriminator loss"""
    def __init__(self, generator, discriminator):
        super(DiscriminatorLoss, self).__init__()
        self.net_G = generator
        self.net_D = discriminator
        self.cyc_loss = P.ReduceMean()
        self.cls_loss = ClassificationLoss()
        self.WGANLoss = WGANGPGradientPenalty(discriminator)

        self.lambda_rec = Tensor(lambda_rec)
        self.lambda_cls = Tensor(lambda_cls)
        self.lambda_gp = Tensor(lambda_gp)

    def construct(self, x_real, c_org, c_trg):
        """Get discriminator loss"""
        # Compute loss with real images
        real_src, real_cls = self.net_D(x_real)

        D_real_loss = - self.cyc_loss(real_src)
        D_real_cls_loss = self.cls_loss(real_cls, c_org)

        # Compute loss with fake images
        x_fake = self.net_G(x_real, c_trg)
        fake_src, _ = self.net_D(x_fake)
        D_fake_loss = self.cyc_loss(fake_src)

        D_gp_loss = self.WGANLoss(x_real, x_fake)

        d_loss = D_real_loss + D_fake_loss + self.lambda_cls * D_real_cls_loss + self.lambda_gp *D_gp_loss

        return (d_loss, D_real_loss, D_fake_loss, D_real_cls_loss, D_gp_loss)
