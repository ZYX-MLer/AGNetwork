# coding=utf-8
import argparse

import torch
import torchvision
import math
import BRSeg


import os
import sys
sys.path.append("../")
import tool.MedicalImagePreprocess as mp
import tool.ToolBox as tBox

import socket
import shutil

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import numpy
import nibabel as nib
nib.Nifti1Header.quaternion_threshold = - numpy.finfo(numpy.float32).eps * 10

import warnings
warnings.filterwarnings("ignore")

import cv2
import medpy.io
def main():

    num_class_local = int(input("num_class_local:"))
    # img, h = medpy.io.load("/lustre/home/yxzhao/Work/Data/BrainRegion/S40/S40.delineation.skullstripped.hdr")
    # for i in img:
    #     cv2.imshow("a", i.astype(numpy.float) / numpy.max(i))
    #     cv2.waitKey()
    hostName = socket.gethostname()

    if hostName.find("ZYX") != -1:
        data_l_path = r"F:\ADNI SkullStripping"
        data_a_path = r"F:\ADNI SkullStripping"
        save_path = r"/lustre/home/yxzhao/Work/Data/ADNI/Result"
        attr_path = r"/lustre/home/yxzhao/Work/Data/ADNI/attribute.csv"

        num_epoch = 100

        num_batch_one_gpu = 6
        channel_rate = 2
        crop_shape = [160, 224, 160]
    elif hostName.find("zyx") != -1:
        train_l_path = r"/Users/zyx/Data/BrainRegion/Train/Raw/training-images"
        train_a_path = r"/Users/zyx/Data/BrainRegion/Train/Aug/AugData"
        test_path = r"/Users/zyx/Data/BrainRegion/Test/testing-images"

        save_path = r"/Users/zyx/Data/BrainRegion/Result"


        num_epoch = 100
        num_batch_one_gpu = 1

        begin_out_channel = 2

        level_skull = 3
        level_whole = 3
        num_split_whole = [2, 2, 2, 1, 1]
    elif hostName.find("node") != -1:

        train_u_path = '/lustre/home/yxzhao/Work/Data/BrainRegion/Diagnosis_AD_Scaled_crop_label/'

        test_path = {}
        if num_class_local == 135:
            train_l_path = '/lustre/home/yxzhao/Work/Data/BrainRegion/Train/Raw'
            test_path["test"] = '/lustre/home/yxzhao/Work/Data/BrainRegion/Test/testing-images'
        else:
            train_l_path = '/lustre/home/yxzhao/Work/Data/BrainRegion/Brain133_nor/Train'
            test_path["test"] = "/lustre/home/yxzhao/Work/Data/BrainRegion/Brain133_nor/Test"
            test_path["ADNI"] = "/lustre/home/yxzhao/Work/Data/BrainRegion/Brain133_nor/ADNI"
            test_path["CADNI"] = "/lustre/home/yxzhao/Work/Data/BrainRegion/Brain133_nor/CADNI"
            test_path["colin27"] = "/lustre/home/yxzhao/Work/Data/BrainRegion/Brain133_nor/colin27"

        save_path = r"/lustre/home/yxzhao/Work/Data/BrainRegion/Result"

        num_epoch = 100
        num_batch_one_gpu = 1

        begin_out_channel = 16

        level_global = 2
        level_local = 0

        img_shape = [180, 223, 252]
        brain_shape = [159, 154, 195]

    parser = argparse.ArgumentParser()
    # IO param
    parser.add_argument("--train_l_path", default=train_l_path)
    parser.add_argument("--train_u_path", default=train_u_path)
    parser.add_argument("--test_path", default=test_path)
    parser.add_argument("--model_path", default=r"/lustre/home/yxzhao/Work/Data/BrainRegion/Result/node11_2_3_79.8/118_79.86_best.tar")
    parser.add_argument("--save_path", default=save_path)

    # Structure Prarm
    parser.add_argument("--begin_in_channel", default=1)
    parser.add_argument("--begin_out_channel", default=begin_out_channel)
    parser.add_argument("--channel_rate", default=1.2)
    parser.add_argument("--d_channel", default=8)

    # Data Param
    parser.add_argument("--num_feature", default=2)

    parser.add_argument("--level_global", default=level_global)
    parser.add_argument("--level_global_mid", default=7)
    # parser.add_argument("--shape_global", default=[180, 228, 252])
    parser.add_argument("--num_class_local", default=num_class_local, type=float)

    if num_class_local == 135:
        parser.add_argument("--shape_global", default=[180 + 1 + 2 + 2 + 2 + 2 + 2 + 2, 223 + 2, 252 + 1 + 2 + 2])
        parser.add_argument("--shape_brain", default=[159 + 2 + 2 + 2 - 4, 154 + 1 + 2 - 4, 195 + 2 - 4])
    else:
        # parser.add_argument("--shape_global", default=[180 + 1 + 2 + 2 + 2 + 2 + 2 + 2, 223 + 2, 252 + 1 + 2 + 2])
        parser.add_argument("--shape_global", default=[207 + 2, 251 + 6, 255 + 2])
        parser.add_argument("--shape_brain", default=[153, 157 + 2 + 2, 191 + 2])
        # parser.add_argument("--shape_global", default=[207 + 2, 251 + 6, 255 + 2])
        # parser.add_argument("--shape_brain", default=[153 + 8, 157 + 2 + 2, 191 + 2])


    # parser.add_argument("--shape_local", default=[133, 133, 161])
    parser.add_argument("--shape_local", default=[117, 117, 117])
    # parser.add_argument("--shape_local_u", default=[133, 133, 161])
    # parser.add_argument("--shape_global", default=[math.ceil(v / (2 ** (level_global + 1))) * (2 ** (level_global + 1)) for v in [180, 223, 252]])
    parser.add_argument("--rate_global", default=0.5, type=float)
    parser.add_argument("--num_class_global", default=1 + 1, type=float)



    parser.add_argument("--level_local", default=level_local)
    # parser.add_argument("--size_local", default=128)
    # parser.add_argument("--shape_local", default=[166, 158, 204])

    # parser.add_argument("--rate_local", default=0.5, type=float)
    parser.add_argument("--kernel_size_head_local", default=5, type=float)
    parser.add_argument("--opt_stride", default=2, type=float)
    parser.add_argument("--opt_stride_u", default=3, type=float)
    parser.add_argument("--num_semi_patch", default=5, type=int)
    # Training Param
    parser.add_argument("--num_epoch", default=num_epoch, type=int)
    parser.add_argument("--num_batch_one_gpu", default=num_batch_one_gpu)
    parser.add_argument("--clip_grad_th_begin", default=0.01, help="0.01")
    parser.add_argument("--clip_grad_th_end", default=0.01, help="0.01")

    parser.add_argument("--ema", default=True)
    parser.add_argument("--ema_rate", default=0.999, type=float)

    parser.add_argument("--learning_rate_begin", default=0.2, type=float, help="0.3")
    parser.add_argument("--learning_rate_param", default=0.9995)
    parser.add_argument("--learning_rate_warmup", default=0.3, type=float)

    parser.add_argument("--lambda_rate_begin", default=0.001, type=float)
    parser.add_argument("--lambda_rate_end", default=0.00001, type=float)
    parser.add_argument("--lambda_rate_param", default=1.5)
    parser.add_argument("--lambda_rate", default=1.0)

    parser.add_argument("--hard_sample_begin", default=2.0)
    parser.add_argument("--hard_sample_end", default=7.0)
    parser.add_argument("--hard_sample_param", default=1.5)
    parser.add_argument("--hard_warm_epoch", default=num_epoch)





    param, unparsed = parser.parse_known_args()





    all_gpu_idx = input("all_gpu_idx(0,1,2,3):")
    os.environ["CUDA_VISIBLE_DEVICES"] = all_gpu_idx
    param.all_gpu_idx = [int(v) for v in all_gpu_idx.split(",")]

    param.num_batch = len(param.all_gpu_idx) * param.num_batch_one_gpu
    # param.num_batch = param.num_batch_one_gpu
    param.num_worker = int(input("num_worker:"))

    step = input("step(train/test):").lower()
    param.step = step

    if step == "train":
        param.save_path = os.path.join(param.save_path, hostName + "_" + "_".join([str(v) for v in param.all_gpu_idx]))


        param.learning_rate_warmup = 0.25
        param.dropout_rate = 0.1
        # param.learning_rate_begin = 0.05
        # param.clip_grad_th_begin = 0.01
        # param.clip_grad_th_end = param.clip_grad_th_begin
        param.max_class_sample = -1


        # param.learning_rate_warmup = float(input("learning_rate_warmup:"))
        # param.dropout_rate = float(input("dropout_rate:"))
        param.angle = 0
        param.learning_rate_begin = float(input("learning_rate_begin:"))
        param.clip_grad_th_begin = float(input("clip_grad_th_begin:"))
        param.clip_grad_th_end = param.clip_grad_th_begin

        param.opt_u = input("opt_u(ce/kl/fce/fkl):").lower().lower()
        param.focal_gamma = float(input("focal_gamma:"))
        # param.focal_gamma_u = float(input("focal_gamma_u:"))
        param.lr_decay = 1
        param.semi_rate = float(input("semi_rate:"))
        param.debug = input("debug(y/n):").lower() == "y"
        # param.using_plabel = input("using_plabel(y/n):").lower() == "y"
        # if param.aug_data is False:
        #     param.num_batch = param.num_batch_one_gpu * len(param.all_gpu_idx)
        # param.max_class_sample = int(input("max_class_sample:"))





        param.mesg = input("描述本次训练的主要目的：")
        d = str(input("新建目录？" + param.save_path + " 目录中之前的数据将被删除（y/n）: "))
        if d.lower() == "y":
            if os.path.exists(param.save_path) == True:
                shutil.rmtree(param.save_path)
            os.makedirs(param.save_path)


        paramFile = open(param.save_path + "/param.txt", "w")
        param_dict = vars(param)
        allKey = sorted(param_dict.keys())
        for k in allKey:
            print(k, ":", param_dict[k])
            s = k + ":" + str(param_dict[k]) + "\n"
            paramFile.write(s)
        paramFile.close()

        srcPath = sys.path[0]
        dstPath = os.path.join(param.save_path, "src")
        if os.path.exists(dstPath) is False:
            shutil.copytree(srcPath, dstPath)


        br_seg = BRSeg.BRSeg(param)
        br_seg.train_model()
    else:
        br_seg = BRSeg.BRSeg(param)
        br_seg.test_model()



if __name__ == '__main__':
    main()