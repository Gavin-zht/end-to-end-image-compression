import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter
from models import *

# zht test first commit
def save_model(model, iter, name):
    """_summary_
    将训练迭代了iter轮的模型model进行保存，保存到name所指明的路径下
    Args:
        model (_type_): 模型对象
        iter (_type_): 训练迭代次数
        name (_type_): 
    """
    torch.save(model.state_dict(), os.path.join(name, "iter_{}.pth.tar".format(iter)))


def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0


class ImageCompressor(nn.Module):
    """
    图像压缩模型
    继承自nn.Module类
    
    """
    def __init__(self, out_channel_N=128):
        super(ImageCompressor, self).__init__()
        self.Encoder = Analysis_net_17(out_channel_N=out_channel_N) # 分析变换网络作为模型的编码器Encoder
        self.Decoder = Synthesis_net_17(out_channel_N=out_channel_N) # 合成变换网络作为模型的解码器Decoder
        self.bitEstimator = BitEstimator(channel=out_channel_N) 
        self.out_channel_N = out_channel_N  # 图像压缩模型的输出通道数

    def forward(self, input_image):
        """_summary_

        Args:
            input_image : 输入图像， 维度为: (batch_size, Channels, height, width)

        Returns:
            _type_: _description_
        """
        quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_N, input_image.size(2) // 16, input_image.size(3) // 16).cuda() 
        #* quant_noise_feature: 一个全零张量，用于存储量化噪声特征，维度为[batch_size, Channels, Height/16, Width/16]
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)  #* 初始化quant_noise_feature为均匀分布的噪声, quant_noise_feature的维度为[batch_size, Channels, Height/16, Width/16]
        feature = self.Encoder(input_image) #* feature 表示 编码器的输出，i.e. 隐变量y
        batch_size = feature.size()[0]  # batch_size 表示 小批量的样本数量
        feature_renorm = feature
        if self.training:
            #* 如果此时模型处于训练状态，那么就用 (feature_renorm + quant_noise_feature) 作为 量化值
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            #* 如果此时模型处于此时状态，那么就用 torch.round(feature_renorm) 作为 量化值
            compressed_feature_renorm = torch.round(feature_renorm)
            
        #* 至此，compressed_feature_renorm 表示 经过量化后的编码值。
        #* 经过量化的编码值被传送给解码器Decoder    
        recon_image = self.Decoder(compressed_feature_renorm) #* 解码器用量化后的编码值作为输入，得到重建图像
        # recon_image = prediction + recon_res
        clipped_recon_image = recon_image.clamp(0., 1.)
        # distortion
        mse_loss = torch.mean((recon_image - input_image).pow(2))   #* 使用MSE计算原始图像和重建图像之间的差距，得到失真distortion

        # def feature_probs_based_sigma(feature, sigma):
        #     mu = torch.zeros_like(sigma)
        #     sigma = sigma.clamp(1e-10, 1e10)
        #     gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        #     probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
        #     total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
        #     return total_bits, probs

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator(z + 0.5) - self.bitEstimator(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob

        total_bits_feature, _ = iclr18_estimate_bits_z(compressed_feature_renorm)
        im_shape = input_image.size()
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])

        return clipped_recon_image, mse_loss, bpp_feature
