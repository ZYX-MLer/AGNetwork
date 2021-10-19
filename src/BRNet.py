import os
import sys
import cv2
sys.path.append("../")
import tool.ToolBox as tBox

import matplotlib
matplotlib.use("TkAgg")

import numpy
import math
import torch
import torchvision

import monai.losses
import monai.metrics
import positional_encodings

import einops

class SEBlock(torch.nn.Module):

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.out_channel = channel
        self.avg_pool = torch.nn.AdaptiveAvgPool3d(1)
        num_dim = max(4, channel // reduction)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, num_dim, bias=False),
            torch.nn.PReLU(num_dim),
            torch.nn.Linear(num_dim, channel, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x).view(x.shape[0], x.shape[1])
        y = self.fc(y).view(x.shape[0], x.shape[1], 1, 1, 1)

        return x * y

class Conv3d(torch.nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, stride=1, att=False):
        super(Conv3d, self).__init__()

        self.all_layer = [] if att is False else [SEBlock(in_channel)]
        self.all_layer = self.all_layer + [
            torch.nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, bias=False, stride=stride),
            # torch.nn.BatchNorm3d(out_channels),
            torch.nn.InstanceNorm3d(out_channel, affine=True),
            # SynchronizedBatchNorm1d(out_channels, eps=1e-5, affine=True),
            torch.nn.PReLU(num_parameters=out_channel)
        ]

        self.all_layer = torch.nn.ModuleList(self.all_layer)
        self.in_channel = in_channel
        self.out_channel = out_channel


    def forward(self, x):
        for layer in self.all_layer:
            x = layer(x)
        return x

class Dense(torch.nn.Module):

    def __init__(self, in_channel, d_channel, kernel_size=3, padding=1, stride=1, att=False):
        super(Dense, self).__init__()
        self.net = Conv3d(in_channel, d_channel, kernel_size=kernel_size, padding=padding, stride=1, att=False)
        self.in_channel = in_channel
        self.out_channel = in_channel + d_channel

    def forward(self, x):

        x_out = self.net(x)
        x = torch.cat([x, x_out], dim=1)
        return x, x_out



class DeConv3d(torch.nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, stride=1, att=False):
        super(DeConv3d, self).__init__()

        self.all_layer = [] if att is False else [SEBlock(in_channel)]
        self.all_layer = self.all_layer + [
            torch.nn.ConvTranspose3d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, bias=False, stride=stride),
            # torch.nn.BatchNorm3d(out_channels),
            torch.nn.InstanceNorm3d(out_channel, affine=True),
            torch.nn.PReLU(num_parameters=out_channel)
        ]

        self.all_layer = torch.nn.ModuleList(self.all_layer)
        self.in_channel = in_channel
        self.out_channel = out_channel


    def forward(self, x):
        for layer in self.all_layer:
            x = layer(x)
        return x

class Conv3dRT(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(Conv3dRT, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.net = torch.nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size)



    def forward(self, x, pad_tpye = "valid", stride = 1, **kwargs):

        if pad_tpye == "valid":
            padding = 0
        elif pad_tpye == "same":
            padding = self.kernel_size // 2

        x = torch.nn.functional.conv3d(x, self.net.weight, self.net.bias, padding=padding, stride=stride)

        # weight_label = torch.zeros([1, 1, self.kernel_size, self.kernel_size, self.kernel_size], dtype=torch.float, device=x.device)
        # weight_label.data[:, :, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2] = 1
        # x = x[:, 0:1]
        # x_1 = torch.nn.functional.conv3d(x, weight_label, padding=padding, stride=stride)
        # x_2 = x[:, :,
        #       self.kernel_size // 2: -self.kernel_size // 2: stride,
        #       self.kernel_size // 2: -self.kernel_size // 2: stride,
        #       self.kernel_size // 2: -self.kernel_size // 2: stride]
        # v = torch.sum(x_1 - x_2)
        return x

class Conv3dRTMulti(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(Conv3dRTMulti, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.net = torch.nn.ModuleList()
        for k in range(kernel_size, -1, -2):
            layer = torch.nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size)
            self.net.append(layer)



    def forward(self, x, pad_tpye = "valid", stride = 1, **kw):

        result = 0
        for layer in self.net:
            kernel = layer.weight.size(-1)
            padding = (self.kernel_size - kernel) // 2
            feature = x[:, :, padding: -padding, padding: -padding, padding: -padding] if padding != 0 else x
            result = result + torch.nn.functional.conv3d(feature, layer.weight, layer.bias, padding=0, stride=stride)

        # weight_label = torch.zeros([1, 1, self.kernel_size, self.kernel_size, self.kernel_size], dtype=torch.float, device=x.device)
        # weight_label.data[:, :, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2] = 1
        # x = x[:, 0:1]
        # x_1 = torch.nn.functional.conv3d(x, weight_label, padding=padding, stride=stride)
        # x_2 = x[:, :,
        #       self.kernel_size // 2: -self.kernel_size // 2: stride,
        #       self.kernel_size // 2: -self.kernel_size // 2: stride,
        #       self.kernel_size // 2: -self.kernel_size // 2: stride]
        # v = torch.sum(x_1 - x_2)
        return result

class Conv3dRTAtt(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, num_conv=1):
        super(Conv3dRTAtt, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.num_conv = num_conv

        self.att_0 = torch.nn.Conv3d(in_channel, num_conv * 2, kernel_size=kernel_size)
        self.att_1 = torch.nn.Sequential(
            torch.nn.PReLU(num_conv * 2),
            torch.nn.Conv3d(num_conv * 2, num_conv, kernel_size=1)
        )

        self.net = torch.nn.Conv3d(in_channel, out_channel * num_conv, kernel_size=kernel_size)



    def forward(self, x, pos, pad_tpye = "valid", stride = 1):

        if pad_tpye == "valid":
            padding = 0
        elif pad_tpye == "same":
            padding = self.kernel_size // 2

        att = torch.nn.functional.conv3d(pos, self.att_0.weight, self.att_0.bias, padding=padding, stride=stride)
        att = self.att_1(att).softmax(dim=1)
        att = att.unsqueeze(2)
        x = torch.nn.functional.conv3d(x, self.net.weight, self.net.bias, padding=padding, stride=stride)
        x = x.view(x.shape[0], self.num_conv, self.out_channel, x.shape[2], x.shape[3], x.shape[4])
        x = torch.sum(x * att, dim=1)
        # weight_label = torch.zeros([1, 1, self.kernel_size, self.kernel_size, self.kernel_size], dtype=torch.float, device=x.device)
        # weight_label.data[:, :, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2] = 1
        # x = x[:, 0:1]
        # x_1 = torch.nn.functional.conv3d(x, weight_label, padding=padding, stride=stride)
        # x_2 = x[:, :,
        #       self.kernel_size // 2: -self.kernel_size // 2: stride,
        #       self.kernel_size // 2: -self.kernel_size // 2: stride,
        #       self.kernel_size // 2: -self.kernel_size // 2: stride]
        # v = torch.sum(x_1 - x_2)
        return x

class UNetMini(torch.nn.Module):

    def __init__(self, in_channel, d_channel, num_level):
        super(UNetMini, self).__init__()

        self.num_level = num_level
        encoder = torch.nn.ModuleList()
        decoder = torch.nn.ModuleList()

        print("mid:", end=">")
        for l_n in range(self.num_level):
            feature = Conv3d(in_channel=in_channel, out_channel=in_channel + d_channel, kernel_size=3, stride=1, padding=0)
            print((in_channel, feature.out_channel), end=">")
            in_channel = feature.out_channel
            encoder.append(feature)

        in_channel = feature.out_channel
        for l_n in range(self.num_level - 1, -1, -1):
            feature = DeConv3d(in_channel=in_channel + encoder[l_n].out_channel,
                               out_channel=encoder[l_n].in_channel, kernel_size=3, stride=1, padding=0)
            print((feature.in_channel, feature.out_channel), end="<")
            in_channel = feature.out_channel
            decoder.append(feature)
        print("mid:", end="<")

        self.net = torch.nn.ModuleDict({
            "encoder": encoder,
            "decoder": decoder
        })

        self.out_channel = feature.out_channel

    def forward(self, feature, dropout_rate = 0):

        all_feature = []
        for l_n in range(self.num_level):
            feature = torch.nn.functional.dropout(feature, p=dropout_rate, training=self.training)
            feature = self.net["encoder"][l_n](feature)
            all_feature.append(feature)

        feature = feature + torch.nn.functional.adaptive_avg_pool3d(feature, (1, 1, 1))

        for l_n in range(self.num_level - 1, -1, -1):
            feature = torch.cat([feature, all_feature[l_n]], dim=1)
            feature = torch.nn.functional.dropout(feature, p=dropout_rate, training=self.training)
            feature = self.net["decoder"][self.num_level - l_n - 1](feature)

        return feature

class Interpolate(torch.nn.Module):

    def __init__(self, scale):
        super(Interpolate, self).__init__()
        self.scale = scale

    def forward(self, x):
        if self.scale == 0.5:
            return torch.nn.functional.max_pool3d(x, 2)
        else:
            return torch.nn.functional.interpolate(x, scale_factor=self.scale)

class BREncoder(torch.nn.Module):

    def __init__(self, in_channel, base_channel, num_feature, all_kernel, all_padding, is_first=True):

        super(BREncoder, self).__init__()

        self.in_channel = in_channel
        self.base_channel = base_channel
        self.num_feature = num_feature

        self.net = torch.nn.ModuleList()
        for e_n in range(len(num_feature)):

            all_feature = torch.nn.ModuleList()
            for f_n in range(num_feature[e_n]):
                feature = Conv3d(in_channel=in_channel, out_channel=in_channel * (e_n != 0 or f_n != 0 or is_first is False) + base_channel,
                                 kernel_size=all_kernel[e_n], stride=1, padding=all_padding[e_n])
                in_channel = feature.out_channel
                all_feature.append(feature)
                print((feature.in_channel, feature.out_channel), end=">")


            pool = Conv3d(in_channel=in_channel, out_channel=in_channel * (e_n != 0 or num_feature[e_n] != 0) + base_channel, kernel_size=3, stride=2, padding=1)
            in_channel = pool.out_channel
            print((pool.in_channel, pool.out_channel), end=">>")

            level = torch.nn.ModuleDict({"pool": pool, "feature": all_feature})
            self.net.append(level)

        self.out_channel = in_channel

    def forward(self, feature, dropout_rate):
        all_feature_encoder = []
        for e_n in range(len(self.num_feature)):

            all_feature = []
            for f_n in range(self.num_feature[e_n]):
                feature = torch.nn.functional.dropout(feature, p=dropout_rate, training=self.training) if dropout_rate != 0 else feature
                feature = self.net[e_n]["feature"][f_n](feature)
                all_feature.append(feature)

            feature = self.net[e_n]["pool"](feature)

            all_feature_encoder.append({"feature": all_feature, "pool": feature})

        return feature, all_feature_encoder

class BRDecoder(torch.nn.Module):

    def __init__(self, in_channel, base_channel, num_feature, in_encoder, all_kernel, all_padding):

        super(BRDecoder, self).__init__()

        self.in_channel = in_channel
        self.all_out_channel = []
        self.base_channel = base_channel
        self.num_feature = num_feature
        self.in_encoder = in_encoder
        self.all_out_channel = []

        self.net = torch.nn.ModuleList()
        for e_n in range(len(num_feature) - 1, -1, -1):

            unpool = DeConv3d(in_channel=in_channel + self.in_encoder.net[e_n]["pool"].out_channel,
                              out_channel=self.in_encoder.net[e_n]["pool"].in_channel, kernel_size=3, stride=2, padding=1)
            in_channel = unpool.out_channel
            print((unpool.in_channel, unpool.out_channel), end=">")

            all_feature = torch.nn.ModuleList()
            num_channel = 0
            for f_n in range(num_feature[e_n] - 1, -1, -1):

                feature = DeConv3d(in_channel=in_channel + self.in_encoder.net[e_n]["feature"][f_n].out_channel,
                                 out_channel=self.in_encoder.net[e_n]["feature"][f_n].in_channel,
                                   kernel_size=all_kernel[e_n], stride=1, padding=all_padding[e_n])
                in_channel = feature.out_channel
                num_channel += in_channel
                all_feature.append(feature)
                print((feature.in_channel, feature.out_channel), end=">")
            print(end=">")
            self.all_out_channel.append(num_channel)
            level = torch.nn.ModuleDict({"unpool": unpool, "feature": all_feature})
            self.net.append(level)

        self.out_channel = in_channel

        print()

    def forward(self, feature, feature_encoder, dropout_rate, all_mask=None):

        all_feature_decoder = []
        all_result_decoder = []
        all_roi_decoder = []
        for d_n in range(len(self.num_feature) - 1, -1, -1):

            feature = torch.cat([feature, feature_encoder[d_n]["pool"]], dim=1)
            pool = self.net[len(self.num_feature) - d_n - 1]["unpool"](feature)
            feature = pool

            all_feature = []
            for f_n in range(self.num_feature[d_n] - 1, -1, -1):
                feature = torch.cat([feature, feature_encoder[d_n]["feature"][f_n]], dim=1)
                feature = torch.nn.functional.dropout(feature, p=dropout_rate, training=self.training) if dropout_rate != 0 else feature
                feature = self.net[len(self.num_feature) - d_n - 1]["feature"][self.num_feature[d_n] - f_n - 1](feature)
                all_feature.append(feature)

            all_feature_decoder.append({"unpool": feature, "feature": all_feature})

        return feature, all_feature_decoder

class BRNet(torch.nn.Module):

    def __init__(self, param):
        super(BRNet, self).__init__()
        self.param = param
        channel_rate = self.param.channel_rate

        in_channel = 1
        out_channel = self.param.begin_out_channel
        print("net_global_brain", end="->")
        begin_global_brain = Conv3d(in_channel=1, out_channel=self.param.begin_out_channel // 4, kernel_size=3, padding=1, stride=2)
        encoder_global_brain = BREncoder(in_channel=begin_global_brain.out_channel,
                                     base_channel=self.param.begin_out_channel // 4, num_feature=[0, 2, 3], all_kernel=[5, 3, 3], all_padding=[0, 0, 0])
        decoder_global_brain = BRDecoder(in_channel=encoder_global_brain.out_channel,
                                     base_channel=self.param.begin_out_channel // 4, num_feature=[0, 2, 3], all_kernel=[5, 3, 3], all_padding=[0, 0, 0], in_encoder=encoder_global_brain)
        head_brain = torch.nn.Conv3d(encoder_global_brain.in_channel + decoder_global_brain.out_channel, self.param.num_class_global, kernel_size=3, padding=1)

        print("net_global_region", end="->")
        begin_global_region = DeConv3d(in_channel=begin_global_brain.out_channel + decoder_global_brain.out_channel,
                                     out_channel=begin_global_brain.out_channel + decoder_global_brain.out_channel, kernel_size=3, padding=1, stride=2)

        encoder_global_region_0 = BREncoder(in_channel=1 + begin_global_region.out_channel,
                                            base_channel=self.param.begin_out_channel // 4 * 3 - 1, num_feature=[1], all_kernel=[5], all_padding=[0])
        encoder_global_region_1 = BREncoder(in_channel=encoder_global_region_0.out_channel,
                                            base_channel=self.param.begin_out_channel // 4 * 3, num_feature=[3], all_kernel=[3], all_padding=[0], is_first=False)
        encoder_global_region_2 = BREncoder(in_channel=encoder_global_region_1.out_channel,
                                            base_channel=self.param.begin_out_channel // 4 * 3, num_feature=[10], all_kernel=[3], all_padding=[0], is_first=False)

        decoder_global_region_2 = BRDecoder(in_channel=encoder_global_region_2.out_channel, base_channel=self.param.begin_out_channel // 4 * 3,
                                            num_feature=[10], all_kernel=[3], all_padding=[0], in_encoder=encoder_global_region_2)

        decoder_global_region_1 = BRDecoder(in_channel=decoder_global_region_2.out_channel, base_channel=self.param.begin_out_channel // 4 * 3,
                                            num_feature=[3], all_kernel=[3], all_padding=[0], in_encoder=encoder_global_region_1)

        decoder_global_region_0 = BRDecoder(in_channel=decoder_global_region_1.out_channel, base_channel=self.param.begin_out_channel // 4 * 3,
                                            num_feature=[1], all_kernel=[5], all_padding=[0], in_encoder=encoder_global_region_0)

        self.pos_encoding = positional_encodings.PositionalEncodingPermute3D(encoder_global_region_0.in_channel * 2)
        self.net = torch.nn.ModuleDict({
            "begin_brain": begin_global_brain,
            "encoder_brain": encoder_global_brain,
            "decoder_brain": decoder_global_brain,
            "head_brain": head_brain,

            "begin_region": begin_global_region,
            "encoder_region_l0": encoder_global_region_0,
            "encoder_region_l1": encoder_global_region_1,
            "encoder_region_l2": encoder_global_region_2,

            "decoder_region_l2": decoder_global_region_2,
            "decoder_region_l1": decoder_global_region_1,
            "decoder_region_l0": decoder_global_region_0,
            # "head_region": Conv3dRTAtt(encoder_global_region_0.in_channel * 2, self.param.num_class_local, kernel_size=self.param.kernel_size_head_local, num_conv=self.param.num_conv)
            "head_region": Conv3dRT(encoder_global_region_0.in_channel * 2, self.param.num_class_local, kernel_size=self.param.kernel_size_head_local)
        })


    def gen_max_roi(self, in_size, crop_size):

        x_begin = int(numpy.random.uniform(0, in_size[0] - crop_size[0]))
        y_begin = int(numpy.random.uniform(0, in_size[1] - crop_size[1]))
        z_begin = int(numpy.random.uniform(0, in_size[2] - crop_size[2]))

        rect = [x_begin, y_begin, z_begin, crop_size[0], crop_size[1], crop_size[2]]

        return rect

    def roi_crop(self, seg, feature_region, pos, padding, num_crop):

        in_size = seg.shape[2:]

        all_seg_local = []
        all_feature_region_local = []
        all_pos_local = []
        for c_n in range(num_crop):
            crop_roi = self.gen_max_roi(in_size=in_size, crop_size=self.param.shape_local)

            seg_local = seg[:, :,
                        crop_roi[0]: crop_roi[0] + crop_roi[3],
                        crop_roi[1]: crop_roi[1] + crop_roi[4],
                        crop_roi[2]: crop_roi[2] + crop_roi[5]
                        ]

            feature_region_local = feature_region[:, :,
                            crop_roi[0]: crop_roi[0] + crop_roi[3] + padding * 2,
                            crop_roi[1]: crop_roi[1] + crop_roi[4] + padding * 2,
                            crop_roi[2]: crop_roi[2] + crop_roi[5] + padding * 2
                            ]

            pos_local = pos[:, :,
                        crop_roi[0]: crop_roi[0] + crop_roi[3] + padding * 2,
                        crop_roi[1]: crop_roi[1] + crop_roi[4] + padding * 2,
                        crop_roi[2]: crop_roi[2] + crop_roi[5] + padding * 2
                        ]
            all_seg_local.append(seg_local)
            all_feature_region_local.append(feature_region_local)
            all_pos_local.append(pos_local)

        seg_local = torch.cat(all_seg_local, dim=0)
        feature_region_local = torch.cat(all_feature_region_local, dim=0)
        pos_local = torch.cat(all_pos_local, dim=0)
        return seg_local, feature_region_local, pos_local

    def split_idx(self, vec):

        num_vec = len(vec)
        all_idx = [] if vec[0] == 0 else [0]
        for i in range(1, num_vec):
            if vec[i - 1] == 0 and vec[i] != 0:
                all_idx.append(i)
            elif vec[i - 1] != 0 and vec[i] == 0:
                all_idx.append(i)
        all_idx = all_idx if vec[-1] == 0 else all_idx + [num_vec]

        assert len(all_idx) % 2 == 0
        all_idx = [[all_idx[i * 2], all_idx[i * 2 + 1]] for i in range(len(all_idx) // 2)]

        all_dif = [v[1] - v[0] for v in all_idx]

        if len(all_dif) == 0:
            return 0, len(vec)
        else:
            return  all_idx[all_dif.index(max(all_dif))]

    def bounding_box(self, object):
        def begin_end(l):
            l_begin = 0
            l_end = len(l)

            for l_n, ll in enumerate(l):
                if ll != 0:
                    l_begin = l_n
                    break

            l = l[::-1]
            for l_n, ll in enumerate(l):
                if ll != 0:
                    l_end = len(l) - l_n
                    break

            return l_begin, l_end

        shape = object.shape
        x = [torch.sum(object[:, 0, v, :, :]).item() for v in range(shape[2])]
        # x_begin, x_end = begin_end(x)
        x_begin, x_end = self.split_idx(x)

        y = [torch.sum(object[:, 0, :, v, :]).item() for v in range(shape[3])]
        # y_begin, y_end = begin_end(y)
        y_begin, y_end = self.split_idx(y)

        z = [torch.sum(object[:, 0, :, :, v]).item() for v in range(shape[4])]
        # z_begin, z_end = begin_end(z)
        z_begin, z_end = self.split_idx(z)

        return {"x": x_begin, "y": y_begin, "z": z_begin, "x_w": x_end - x_begin, "y_w": y_end - y_begin, "z_w": z_end - z_begin}

    def box_inside(self, box_small, box_big):
        box_small_begin = [box_small["x"], box_small["y"], box_small["z"]]
        box_small_end = [box_small["x"] + box_small["x_w"], box_small["y"] + box_small["y_w"], box_small["z"] + box_small["z_w"]]

        box_big_begin = [box_big["x"], box_big["y"], box_big["z"]]
        box_big_end = [box_big["x"] + box_big["x_w"], box_big["y"] + box_big["y_w"], box_big["z"] + box_big["z_w"]]

        cond = 0
        cond += box_big_end[0] >= box_small_begin[0] >= box_big_begin[0]
        cond += box_big_end[1] >= box_small_begin[1] >= box_big_begin[1]
        cond += box_big_end[2] >= box_small_begin[2] >= box_big_begin[2]
        cond += box_big_end[0] >= box_small_end[0] >= box_big_begin[0]
        cond += box_big_end[1] >= box_small_end[1] >= box_big_begin[1]
        cond += box_big_end[2] >= box_small_end[2] >= box_big_begin[2]

        return cond == 6

    def exp_mask_box(self, mask_box, img_shape, crop_size):

        mask_box_exp = {}

        d_x_w = crop_size[0] - mask_box["x_w"]
        d_x = int(numpy.random.uniform(d_x_w // 4, d_x_w // 2 + 0.1)) if self.training is True else d_x_w // 2
        mask_box_exp["x"] = min(max(0, mask_box["x"] - d_x), img_shape[2] - crop_size[0])
        mask_box_exp["x_w"] = crop_size[0]

        d_y_w = crop_size[1] - mask_box["y_w"]
        d_y = int(numpy.random.uniform(d_y_w // 4, d_y_w // 2 + 0.1)) if self.training is True else d_y_w // 2
        mask_box_exp["y"] = min(max(0, mask_box["y"] - d_y), img_shape[3] - crop_size[1])
        mask_box_exp["y_w"] = crop_size[1]

        d_z_w = crop_size[2] - mask_box["z_w"]
        d_z = int(numpy.random.uniform(d_z_w // 4, d_z_w // 2 + 0.1)) if self.training is True else d_z_w // 2
        mask_box_exp["z"] = min(max(0, mask_box["z"] - d_z), img_shape[4] - crop_size[2])
        mask_box_exp["z_w"] = crop_size[2]

        return mask_box_exp


    def crop_mask(self, mask, crop_size, margin=1, target=None):


        mask_box = self.bounding_box(mask)
        mask_box_exp = self.exp_mask_box(mask_box, mask.shape, crop_size)
        target_box = self.bounding_box(target) if target is not None else None


        if self.box_inside(box_small=target_box, box_big=mask_box_exp) is False:
            # print(target_box, mask_box_exp, "in")
            return None, None

        if mask_box_exp["x_w"] > crop_size[0] or mask_box_exp["y_w"] > crop_size[1] or mask_box_exp["z_w"] > crop_size[2]:
            # print(target_box, mask_box_exp, "sm")
            return None, None

        roi_2 = mask_box_exp
        roi = {
            "x": mask_box_exp["x"] * 2,
            "y": mask_box_exp["y"] * 2,
            "z": mask_box_exp["z"] * 2,
            "x_w": crop_size[0] * 2 - 1,
            "y_w": crop_size[1] * 2 - 1,
            "z_w": crop_size[2] * 2 - 1}

        return roi, roi_2


    def forward(self, sample, e_n, dropout_rate, type):


        if type == "unlabel_1":
            sample["counter_local"] = self.param.self_flip.mirror_result(sample["counter_local"].detach(), dim=2)
            sample["counter_global"] = sample["counter_global"].flip(dims=[2])

            loss_g, loss_dict_g = self.loss_u(sample["result_global"], sample["counter_global"], e_n, "_g", self.param.num_class_global)
            loss_l, loss_dict_l = self.loss_u(sample["result_local"], sample["counter_local"], e_n, "_l", self.param.num_class_local)

            loss = loss_g + loss_l
            loss_dict = {}
            loss_dict["l_g"] = loss_g
            loss_dict.update(loss_dict_l)

            return loss, loss_dict



        predict = {}

        feature_brain_begin = self.net["begin_brain"](sample["data"])
        feature, all_feature_encoder_brain = self.net["encoder_brain"](feature_brain_begin, dropout_rate=dropout_rate)
        feature_brain, all_feature_decoder_brain = self.net["decoder_brain"](feature=feature, feature_encoder=all_feature_encoder_brain, dropout_rate=dropout_rate)
        head_brain = self.net["head_brain"](torch.cat([feature_brain, feature_brain_begin], dim=1))
        predict["global"] = head_brain


        # mask = torch.topk(head_brain, dim=1, k=1)[1] if e_n > self.param.pre_epoch * 3 else sample["seg_global"]
        # shape_brain_s = [(v + 1) // 2 for v in self.param.shape_brain]
        # roi, roi_2 = self.crop_mask(mask, [(v + 1) // 2 for v in self.param.shape_brain])

        if "seg_global" in sample:
            mask = torch.topk(head_brain, dim=1, k=1)[1]
            roi, roi_2 = self.crop_mask(mask, [(v + 1) // 2 for v in self.param.shape_brain], target=sample["seg_global"])
            if roi is None:
                roi, roi_2 = self.crop_mask(sample["seg_global"], [(v + 1) // 2 for v in self.param.shape_brain], target=sample["seg_global"])
                failed = torch.tensor([1.], device=mask.device)
            else:
                failed = torch.tensor([0.], device=mask.device)
        else:
            roi_ = sample["roi_brain"].detach().cpu().numpy()[0]

            roi = {
                "x": roi_[0], "y": roi_[1], "z": roi_[2],
                "x_w": roi_[3], "y_w": roi_[4], "z_w": roi_[5]
            }

            roi_2 = {
                "x": roi["x"] // 2, "y": roi["y"] // 2, "z": roi["z"] // 2,
                "x_w": (roi["x_w"] + 1) // 2, "y_w": (roi["y_w"] + 1) // 2, "z_w": (roi["z_w"] + 1) // 2
            }
            # roi = {
            #     "x": mask_box_exp["x"] * 2,
            #     "y": mask_box_exp["y"] * 2,
            #     "z": mask_box_exp["z"] * 2,
            #     "x_w": crop_size[0] * 2 - 1,
            #     "y_w": crop_size[1] * 2 - 1,
            #     "z_w": crop_size[2] * 2 - 1}

        feature_brain = torch.cat([feature_brain_begin, feature_brain], dim=1)[:, :,
                        roi_2["x"]: roi_2["x"] + roi_2["x_w"],
                        roi_2["y"]: roi_2["y"] + roi_2["y_w"],
                        roi_2["z"]: roi_2["z"] + roi_2["z_w"]]

        feature_region = self.net["begin_region"](feature_brain)

        feature_region_begin = torch.cat([feature_region,
                             sample["data"][:, :,
                             roi["x"]: roi["x"] + roi["x_w"],
                             roi["y"]: roi["y"] + roi["y_w"],
                             roi["z"]: roi["z"] + roi["z_w"]]], dim=1)

        dr = 5 if type == "unlabel_0" else 1
        feature_encoder_0, all_feature_encoder_region_0 = self.net["encoder_region_l0"](feature_region_begin, dropout_rate=0)
        feature_encoder_1, all_feature_encoder_region_1 = self.net["encoder_region_l1"](feature_encoder_0, dropout_rate=0)
        feature_encoder_2, all_feature_encoder_region_2 = self.net["encoder_region_l2"](feature_encoder_1, dropout_rate=dropout_rate * dr)

        feature_decoder_2, all_feature_decoder_region_2 = self.net["decoder_region_l2"](feature=feature_encoder_2, feature_encoder=all_feature_encoder_region_2, dropout_rate=dropout_rate * dr)
        feature_decoder_1, all_feature_decoder_region_1 = self.net["decoder_region_l1"](feature=feature_decoder_2, feature_encoder=all_feature_encoder_region_1, dropout_rate=0)
        feature_decoder_0, all_feature_decoder_region_0 = self.net["decoder_region_l0"](feature=feature_decoder_1, feature_encoder=all_feature_encoder_region_0, dropout_rate=0)


        feature_region = torch.cat([feature_decoder_0, feature_region_begin], dim=1)
        head_pad = self.param.kernel_size_head_local // 2
        feature_region = torch.nn.functional.pad(feature_region, (head_pad, head_pad, head_pad, head_pad, head_pad, head_pad))
        pos = self.pos_encoding(feature_region)
        feature_region = feature_region + pos


        margin = self.param.level_local
        if type == "label":
            seg_brain = sample["seg"][:, :, roi["x"]: roi["x"] + roi["x_w"], roi["y"]: roi["y"] + roi["y_w"], roi["z"]: roi["z"] + roi["z_w"]]
            seg_local, feature_region_local, pos_local = self.roi_crop(seg_brain, feature_region, pos, padding=head_pad, num_crop=2)
            predict["local"] = self.net["head_region"](feature_region_local, pos=pos_local, stride=self.param.opt_stride)
            predict["seg_local"] = seg_local[:, :, ::self.param.opt_stride, ::self.param.opt_stride, ::self.param.opt_stride]

            sample["predict"] = predict
            loss_opt, loss_dict, dice_weight = self.loss(sample, e_n)
            loss_dict["failed"] = failed

            return {"loss_opt": loss_opt, "loss_dict": loss_dict, "dice_weight": dice_weight}
        elif type == "unlabel_0":

            # f1 = feature_region[:, :, ::self.param.opt_stride, ::self.param.opt_stride, ::self.param.opt_stride]
            # f2 = feature_region.flip(dims=[2])[:, :, ::self.param.opt_stride, ::self.param.opt_stride, ::self.param.opt_stride].flip(dims=[2])
            # v = torch.sum(torch.abs(f1 - f2))
            roi_local = sample["roi_local"].detach().cpu().numpy()[0]
            feature_region_local = feature_region[:, :,
            roi_local[0] + head_pad: roi_local[0] + head_pad + roi_local[3],
            roi_local[1] + head_pad: roi_local[1] + head_pad + roi_local[4],
            roi_local[2] + head_pad: roi_local[2] + head_pad + roi_local[5]
            ]

            result_local = self.net["head_region"](feature_region_local, pos=pos, stride=self.param.opt_stride)
            return head_brain, result_local
        else:

            all_dice_local, min_dice, min_dice_idx, max_dice, max_dice_idx, result = self.loss_measure_local(feature_region, margin=margin + self.param.kernel_size_head_local // 2, pos=pos,
                                                             seg_label=sample["seg"][:, :, roi["x"]: roi["x"] + roi["x_w"], roi["y"]: roi["y"] + roi["y_w"], roi["z"]: roi["z"] + roi["z_w"]],
                                                             output_result=True if self.param.step == "test" else False)
            sample["predict"] = predict
            # all_dice_global = self.loss_measure_global(sample)
            loss_dict = {
                "dice_local": all_dice_local,
                "min_dice": min_dice,
                "max_dice": max_dice,
                "min_dice_idx": min_dice_idx,
                "max_dice_idx": max_dice_idx,
                "failed": [failed]
            }
            # loss_dict = self.loss_measure(sample, feature.device)
            return loss_dict


    def label2oneHot(self, label, num_class):
        shape = list(label.shape)
        shape_onehot = shape
        shape_onehot[1] = num_class

        label_onehot = torch.zeros(shape, dtype=label.dtype, device=label.device)
        label_onehot = label_onehot.scatter_(1, label.long(), 1)
        return label_onehot

    def loss(self, sample, idx_epoch = 0):
        def dice(result, label):
            num_class = result.size(1)
            loss = 0
            voxel_pre = torch.sum(result, dim=[0, 2, 3, 4])

            for c_n in range(num_class):
                cond = (label == c_n)
                voxel_lab = torch.sum(cond, dim=[0, 2, 3, 4])
                voxel_tru = torch.sum(result[:, c_n: c_n + 1][cond])
                loss = loss + (1. - voxel_tru * 2 / (voxel_lab + voxel_pre[c_n]))
            loss = loss / num_class
            return loss

        loss = 0
        loss_dict = {}

        gamma = 1 + min(idx_epoch / self.param.original_num_epoch, 0) * self.param.focal_gamma
        FocalLoss = monai.losses.FocalLoss(gamma=gamma)
        DiceLoss = monai.losses.DiceLoss(softmax=True)

        seg_global = sample["seg_global"]
        result_global = sample["predict"]["global"]
        focal_global = FocalLoss(result_global, seg_global)
        dice_global = DiceLoss(result_global, self.label2oneHot(seg_global, self.param.num_class_global))
        loss = loss + (focal_global + dice_global) * 0.5
        loss_dict["f_g"] = focal_global
        loss_dict["d_g"] = dice_global

        seg_local = sample["predict"]["seg_local"]
        result_local = sample["predict"]["local"]
        focal_local = FocalLoss(result_local, seg_local.contiguous())
        dice_local = monai.losses.DiceLoss(softmax=True, reduction='none')(result_local, self.label2oneHot(seg_local, self.param.num_class_local))
        dice_weight = dice_local.detach()
        dice_local = dice_local.mean()
        loss = loss + (focal_local + dice_local) * 0.5
        loss_dict["f_l"] = focal_local
        loss_dict["d_l"] = dice_local



        return loss, loss_dict, dice_weight

    def loss_u(self, result_map, counter_map, idx_epoch = 0, suffix = "", num_class=1):
        loss = 0
        loss_dict = {}

        result_s0 = result_map
        result_s1 = counter_map
        result_tt = (result_s0.softmax(dim=1) + result_s1.softmax(dim=1)) * 0.5

        result_tt = einops.rearrange(result_tt, "n c x y z -> (n x y z) c")
        result_s0 = einops.rearrange(result_s0, "n c x y z -> (n x y z) c")
        result_s1 = einops.rearrange(result_s1, "n c x y z -> (n x y z) c")

        v_tt, idx_tt = torch.topk(result_tt, dim=1, k=1)
        v_s0, idx_s0 = torch.topk(result_s0, dim=1, k=1)
        v_s1, idx_s1 = torch.topk(result_s1, dim=1, k=1)

        cate_set = set(list(idx_tt.view(-1).detach().cpu().numpy()))
        pre_c = torch.tensor(float(len(cate_set)), device=result_tt.device) / num_class
        loss_dict["n_c" + suffix] = pre_c

        if 0.9 < pre_c < 1:
            cate_set_inv = set(range(num_class)).difference(cate_set)
            for r in list(cate_set_inv):

                idx_tt[idx_s0 == r] = r
                idx_tt[idx_s1 == r] = r

                result_tt[idx_s0.view(-1) == r] = result_s0[idx_s0.view(-1) == r].softmax(dim=1)
                result_tt[idx_s1.view(-1) == r] = result_s1[idx_s1.view(-1) == r].softmax(dim=1)

        cate_set = set(list(idx_tt.view(-1).detach().cpu().numpy()))
        loss_dict["n_c_ed" + suffix] = torch.tensor(float(len(cate_set)), device=result_tt.device) / num_class

        mask = idx_s0 != idx_s1
        loss_dict["n_opt" + suffix] = torch.count_nonzero(mask).float()
        mask = torch.ones_like(mask) if loss_dict["n_opt" + suffix] == 0 else mask

        if self.param.opt_u == "kl":
            err_kl = torch.nn.functional.kl_div(torch.log_softmax(result_s0, dim=1), result_tt, reduction='none').sum(dim=1)
            err_th = err_kl[mask.view(-1)].min()
            err_mask = err_kl > err_th
            err_kl_m = err_kl[err_mask].mean()
            loss_dict["kl_u" + suffix] = err_kl_m
            loss = loss + err_kl_m
        elif self.param.opt_u == "ce":
            err_ce = torch.nn.functional.cross_entropy(result_s0, idx_tt.view(-1), reduction='none').view(-1)
            err_th = err_ce[mask.view(-1)].min()
            err_mask = err_ce > err_th
            err_ce_m = err_ce[err_mask].mean()
            loss_dict["ce_u" + suffix] = err_ce_m
            loss = loss + err_ce_m
        elif self.param.opt_u == "fkl":
            gamma = 1 + min(idx_epoch / self.param.original_num_epoch, 0) * self.param.focal_gamma
            FocalLoss = monai.losses.FocalLoss(gamma=gamma)
            err_fkl = FocalLoss(result_s0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), result_tt.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
            loss_dict["fkl_u" + suffix] = err_fkl
            loss = loss + err_fkl
        elif self.param.opt_u == "fce":
            gamma = 1 + min(idx_epoch / self.param.original_num_epoch, 0) * self.param.focal_gamma
            FocalLoss = monai.losses.FocalLoss(gamma=gamma)
            err_fce = FocalLoss(result_s0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), idx_tt.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
            loss_dict["fce_u" + suffix] = err_fce
            loss = loss + err_fce
        else:
            assert 0

        return loss, loss_dict

    def loss_measure_local(self, feature_in, margin, pos, seg_label, output_result=False):

        num_split = 2
        raw_shape = feature_in.shape
        all_x_loc = [0, raw_shape[2]]
        all_y_loc = [0, raw_shape[3] // num_split, raw_shape[3]]
        all_z_loc = [0, raw_shape[4] // num_split, raw_shape[4]]

        # feature_in = torch.nn.functional.pad(feature_in, (0, 1, 0, 1))
        # seg_label = torch.nn.functional.pad(seg_label, (0, 1, 0, 1))

        all_dice_true = 0
        all_dice_gt = 0
        all_dice_pre = 0
        all_result_x = []
        for x_n in range(1):
            all_result_y = []
            for y_n in range(num_split):
                all_result_z = []
                for z_n in range(num_split):
                    data_dst = torch.zeros([raw_shape[0], raw_shape[1],
                                            all_x_loc[x_n + 1] - all_x_loc[x_n] + margin * 2,
                                            all_y_loc[y_n + 1] - all_y_loc[y_n] + margin * 2,
                                            all_z_loc[z_n + 1] - all_z_loc[z_n] + margin * 2],
                                           device=feature_in.device)
                    x_begin = all_x_loc[x_n]
                    y_begin = all_y_loc[y_n]
                    z_begin = all_z_loc[z_n]
                    x_end = all_x_loc[x_n + 1]
                    y_end = all_y_loc[y_n + 1]
                    z_end = all_z_loc[z_n + 1]


                    feature_crop = feature_in[:, :,
                              x_begin: x_end + self.param.kernel_size_head_local // 2 * 2,
                              y_begin: y_end + self.param.kernel_size_head_local // 2 * 2,
                              z_begin: z_end + self.param.kernel_size_head_local // 2 * 2]

                    pos_crop = pos[:, :,
                              x_begin: x_end + self.param.kernel_size_head_local // 2 * 2,
                              y_begin: y_end + self.param.kernel_size_head_local // 2 * 2,
                              z_begin: z_end + self.param.kernel_size_head_local // 2 * 2]

                    result_patch = self.net["head_region"](feature_crop, pos=pos_crop, stride=1)

                    seg_patch = seg_label[:, :, x_begin: x_end, y_begin: y_end, z_begin: z_end]
                    result_patch_onehot = self.label2oneHot(torch.topk(result_patch, k=1, dim=1)[1], self.param.num_class_local)
                    seg_patch_onehot = self.label2oneHot(seg_patch, self.param.num_class_local)
                    all_dice_gt = all_dice_gt + torch.sum(seg_patch_onehot, dim=[2, 3, 4])
                    all_dice_pre = all_dice_pre + torch.sum(result_patch_onehot, dim=[2, 3, 4])
                    all_dice_true = all_dice_true + torch.sum(seg_patch_onehot * result_patch_onehot, dim=[2, 3, 4])
                    if output_result is True:
                        all_result_z.append(result_patch.detach().cpu())
                if output_result is True:
                    all_result_y.append(torch.cat(all_result_z, dim=-1))
            if output_result is True:
                all_result_x.append(torch.cat(all_result_y, dim=-2))
        if output_result is True:
            result = torch.cat(all_result_x, dim=-3)
            result = torch.topk(result, dim=1, k=1)[1]
        else:
            result = torch.zeros(0, dtype=feature_in.dtype, device=feature_in.device)

        if output_result is True:
            result = result.squeeze(0).squeeze(0).detach().cpu().numpy()
            seg_label = seg_label.squeeze(0).squeeze(0).detach().cpu().numpy()
            for i in range(1, self.param.num_class_local):
                for rr, ss in zip(result, seg_label):
                    r = rr.copy()
                    s = ss.copy()
                    r[r!= i] = 0
                    s[s!= i] = 0

                    d = r - s
                    if numpy.sum(d) == 0:
                        continue

                    max_value = numpy.max(d)
                    min_value = numpy.min(d)
                    d = (d - min_value) / (max_value - min_value) * 255
                    d = d.astype(numpy.uint8)
                    d = cv2.applyColorMap(d, cv2.COLORMAP_JET)
                    # d[d > 0] = 1
                    # d[d < 0] = 0.5

                    cv2.imshow("d", d.astype(numpy.float) / numpy.max(d))

                    cv2.imshow("r", r.astype(numpy.float) / numpy.max(r))
                    cv2.imshow("s", s.astype(numpy.float) / numpy.max(s))

                    cv2.waitKey()

        all_dice_local = 2 * all_dice_true / (all_dice_gt + all_dice_pre + 0.0001)
        max_dice, max_dice_idx = torch.max(all_dice_local[:, :], dim=1)
        min_dice, min_dice_idx = torch.min(all_dice_local[:, :], dim=1)

        max_dice = [max_dice]
        max_dice_idx = [max_dice_idx]
        min_dice = [min_dice]
        min_dice_idx = [min_dice_idx]

        mean_dice = list(torch.mean(all_dice_local[:, 1:], dim=1))
        return mean_dice, min_dice, min_dice_idx, max_dice, max_dice_idx, result
    compare_key = "dice_local"