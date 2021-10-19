import torch
import torch.utils.data
import enum
import os
import sys
import cv2
import numpy
import shutil
import random
random.seed(0)
from random import choice

sys.path.append("../")
import tool.ToolBox as tBox
import tool.MedicalImagePreprocess as mp
import SegToolBox as stBox

import nibabel as nib
nib.Nifti1Header.quaternion_threshold = - numpy.finfo(numpy.float32).eps * 10
from scipy.ndimage import interpolation
import scipy.io
import scipy.ndimage
from scipy.io import loadmat

# 这里两类样本无差别对待
class BRDataSet(torch.utils.data.Dataset):

    area = {
        0: 2000.0329, 1: 3.99259, 7: 2.2857, 4: 3.54864, 11: 2.87875, 3: 2.98059,
        2: 3.35201, 8: 2.18534, 5: 2.55912, 9: 2.09865, 13: 1.91389, 14: 1.86491,
        6: 2.40729, 17: 1.76138, 16: 1.79175, 26: 1.65745, 25: 1.66073, 34: 2.05775,
        51: 21.6868, 30: 1.86618, 41: 9.9814, 66: 2.49694, 43: 1.87185, 49: 6.36534,
        74: 2.26447, 46: 8.38177, 27: 1.65905, 65: 2.51076, 83: 2.18013, 60: 7.99015,
        18: 1.73568, 75: 2.36757, 87: 2.19207, 91: 2.2297, 95: 2.29636, 86: 2.18933,
        96: 2.32561, 99: 2.47508, 103: 4.53115, 68: 2.48432, 97: 2.35646, 104: 3.3096,
        108: 3.56397, 111: 2.48107, 69: 2.49886, 116: 3.1128, 20: 1.69195, 107: 5.52059,
        124: 3.20484, 56: 4.579, 118: 3.00275, 50: 6.62068, 114: 3.99829, 48: 8.19835,
        38: 37.06274, 12: 1.96995, 33: 1.90488, 10: 2.03047, 28: 1.66635, 24: 1.6688,
        39: 38.747, 77: 2.1992, 44: 140.29657, 32: 2.33577, 63: 2.55397, 122: 5.6922,
        58: 5.43594, 15: 1.82479, 19: 1.71447, 21: 1.67695, 123: 5.59386, 130: 2.5738,
        45: 141.65168, 85: 2.18359, 121: 3.50435, 135: 5.355, 139: 4.40781, 40: 12.4297,
        129: 5.18015, 133: 7.33642, 136: 2.9407, 140: 3.01684, 53: 3.70447, 22: 1.66495,
        110: 2.39018, 137: 3.05033, 141: 3.33301, 145: 4.74048, 128: 5.78933, 55: 4.46507,
        42: 1.68679, 117: 3.13377, 149: 2.2026, 90: 2.21649, 29: 1.68036, 105: 3.35893,
        93: 2.25809, 35: 15.07397, 112: 4.07154, 154: 7.45513, 36: 3.78547, 76: 2.34839,
        134: 5.42424, 126: 2.45371, 82: 2.18121, 62: 5.70062, 67: 2.48434, 71: 4.98855,
        79: 2.18902, 148: 2.14557, 131: 2.77053, 84: 2.18334, 100: 4.07844, 132: 8.00776,
        78: 2.19248, 70: 2.55294, 146: 3.28315, 109: 3.69703, 80: 2.18569, 115: 3.87099,
        120: 3.48156, 23: 1.81827, 125: 3.40543, 143: 9.19412, 59: 7.65975, 119: 2.88136,
        57: 5.70811, 144: 4.42748, 64: 2.52969, 113: 3.94807, 152: 4.30212, 61: 5.99237,
        54: 3.67297, 37: 4.46276, 31: 2.26056, 73: 3.70443, 98: 2.39881, 106: 5.61856,
        151: 2.90453, 150: 2.84827, 153: 4.22865, 72: 3.47207, 92: 2.2422, 147: 3.50229,
        142: 9.5112, 94: 2.27685, 102: 4.5428, 138: 4.42131, 52: 18.35984, 155: 7.76191,
        88: 2.19908, 193: 3.64353, 198: 3.88809, 47: 8.57259, 199: 3.62469, 200: 3.20217,
        127: 2.57185, 191: 7.2516, 160: 3.22762, 167: 3.79747, 190: 7.18406, 89: 2.20656,
        192: 3.82795, 101: 4.10541, 185: 1.46529, 171: 2.33877, 188: 1.1714, 176: 4.41927,
        168: 5.53987, 197: 2.25235, 178: 2.33991, 162: 2.74149, 194: 3.04981, 187: 1.48572,
        175: 2.2107, 161: 3.24276, 158: 1.60318, 164: 2.10307, 169: 5.26458, 157: 3.05679,
        189: 1.34799, 183: 5.60902, 184: 1.49088, 165: 2.0265, 173: 2.46037, 81: 2.18321,
        174: 2.08884, 156: 2.9178, 172: 2.34777, 186: 1.4228, 195: 3.49164, 166: 3.49066,
        180: 2.04758, 179: 2.46402, 181: 2.2465, 177: 4.07898, 182: 6.2388, 196: 1.93391,
        163: 2.5863, 159: 1.69448, 170: 2.20801, 206: 0.38066, 201: 3.28584, 202: 4.51007,
        203: 4.44962, 204: 1.24343, 205: 1.33117, 207: 0.29397}

    ignore_list_135 = [
        "OAS1_0101", "OAS1_0111", "OAS1_0117", "OAS1_0379", "OAS1_0395",
        "OAS1_0091", "OAS1_0417", "OAS1_0040", "OAS1_0282", "OAS1_0331",
        "OAS1_0456", "OAS1_0300", "OAS1_0220", "OAS1_0113", "OAS1_0083"
    ]

    ignore_list_133 = ['OAS1_0111_MR1', 'OAS1_0353_MR2', 'OAS1_0032_MR1', 'OAS1_0379_MR1', 'OAS1_0255_MR1']

    def __init__(self, param, data_path, ignore_key=[], transform=None, name="", times=1):
        self.param = param
        self.ignore_list = self.ignore_list_133 if self.param.num_class_local == 133 else self.ignore_list_135
        self.data_path = data_path
        self.name = name

        self.all_data = [v for v in tBox.FindAllFile(data_path, [], ["gz"]) if v.find("_glm.nii.gz") == -1 and v.find("_pglm.nii.gz") == -1]
        self.all_data = [v for v in self.all_data if sum([v.find(vv) for vv in ignore_key]) == -len(ignore_key)]
        # t = [v for v in self.all_data if sum([v.find(vv) for vv in self.ignore_list]) != -len(self.ignore_list)]
        self.all_data = [v for v in self.all_data if sum([v.find(vv) for vv in self.ignore_list]) == -len(self.ignore_list)]
        self.all_data = self.all_data * times
        self.transform = transform

        print(name, "num_data:", len(self.all_data))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):

        idx = item

        while True:
            file_name = self.all_data[idx]

            img = nib.load(file_name).get_data()
            img = img.astype(numpy.float32) / numpy.max(img)
            sample = {"data": img, "tr_flag": True}

            seg_name = file_name.replace("images", "labels").replace(".nii.gz", "_glm.nii.gz")
            if os.path.exists(seg_name) is True:
                seg = nib.load(seg_name).get_data()
                sample["seg"] = seg
            else:
                seg_name = file_name.replace("images", "labels").replace(".nii.gz", "_pglm.nii.gz")
                seg = nib.load(seg_name).get_data()
                sample["seg"] = seg

            if self.transform:
                sample = self.transform(sample)

            if sample["tr_flag"] is True:
                sample.pop("tr_flag")
                break
            else:
                idx = numpy.random.randint(0, len(self.all_data))
                print("@", end="")

        if sample["data"].ndim == 3:
            seg_global = sample["seg"]
            seg_global = interpolation.zoom(seg_global, [(v + 1) / 2 / v for v in seg_global.shape], order=0)
            seg_global[seg_global != 0] = 1

            sample["data"] = numpy.expand_dims(sample["data"], 0)
            sample["seg"] = numpy.expand_dims(sample["seg"], 0).astype(numpy.uint8)
            sample["seg_global"] = numpy.expand_dims(seg_global, 0).astype(numpy.uint8)
            # sample["roi"] = numpy.expand_dims(sample["roi"], 0)
            # sample["data_global"] = numpy.expand_dims(sample["data_global"], 0)
            # sample["seg_global"] = numpy.expand_dims(sample["seg_global"], 0)

            # sample["data_mid"] = numpy.expand_dims(sample["data_mid"], 0)
            # sample["seg_mid"] = numpy.expand_dims(sample["seg_mid"], 0)



        return sample


class BRDataLoaderCompose():
    def __init__(self, all_loader, num_core):
        self.num_core = num_core
        self.all_loader = all_loader
        self.min_iter = min([len(loader) for loader in self.all_loader])
        self.all_iter = [iter(loader) for loader in self.all_loader]

    def __len__(self):
        return self.min_iter

    def __iter__(self):
        return self

    def __next__(self):
        return self.get()

    def get(self):

        try:
            if self.num_core == 1:
                idx = int(numpy.random.uniform(0, len(self.all_iter)))
                all_sample = [next(self.all_iter[idx])]
            else:
                all_sample = [next(iter) for iter in self.all_iter]

            sample = {}
            for key in all_sample[0]:
                data = [s[key] for s in all_sample]
                data = torch.cat(data, dim=0)
                sample[key] = data
        except StopIteration:
            self.all_iter = [iter(loader) for loader in self.all_loader]
            raise StopIteration
        # return sample
        #     data = next(self.iter)
        # except StopIteration:
        #     self.iter = iter(self.dataloader)
        #     data = next(self.iter)

        return sample

class BRDataLoaderEndless():
    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(loader)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        return self.get()

    def get(self):

        try:
            sample = next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            sample = next(self.iter)

        sample.pop("seg")
        sample["data"] = sample["data"].transpose(0, 1)
        sample["roi_local"] = sample["roi_local"].squeeze(0)
        sample["roi_brain"] = sample["roi_brain"].squeeze(0)
        return sample