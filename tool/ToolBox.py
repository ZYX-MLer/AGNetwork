import os
import numpy
# import tensorflow as tf
import random
import cv2
import threading
import math
import csv


def crop_norm(data, roi, shape):
    # assert roi[3] < shape[0] and roi[4] < shape[1] and roi[5] < shape[2]

    data_nor = numpy.zeros(shape, dtype=data.dtype)

    begin_0 = (shape[0] - roi[3]) // 2
    end_0 = begin_0 + roi[3]
    begin_1 = (shape[1] - roi[4]) // 2
    end_1 = begin_1 + roi[4]
    begin_2 = (shape[2] - roi[5]) // 2
    end_2 = begin_2 + roi[5]

    data_nor[
    begin_0: end_0,
    begin_1: end_1,
    begin_2: end_2] = \
        data[
        roi[0]: roi[0] + roi[3],
        roi[1]: roi[1] + roi[4],
        roi[2]: roi[2] + roi[5]]

    roi_norm = [begin_0, begin_1, begin_2, roi[3], roi[4], roi[5]]
    return data_nor, roi_norm


def crop_norm_4(data, roi, shape):
    # assert roi[3] < shape[0] and roi[4] < shape[1] and roi[5] < shape[2]

    data_nor = numpy.zeros([data.shape[0]] + list(shape), dtype=data.dtype)

    begin_0 = (shape[0] - roi[3]) // 2
    end_0 = begin_0 + roi[3]
    begin_1 = (shape[1] - roi[4]) // 2
    end_1 = begin_1 + roi[4]
    begin_2 = (shape[2] - roi[5]) // 2
    end_2 = begin_2 + roi[5]

    data_nor[:,
    begin_0: end_0,
    begin_1: end_1,
    begin_2: end_2] = \
        data[:,
        roi[0]: roi[0] + roi[3],
        roi[1]: roi[1] + roi[4],
        roi[2]: roi[2] + roi[5]]

    roi_norm = [begin_0, begin_1, begin_2, roi[3], roi[4], roi[5]]
    return data_nor, roi_norm

def crop_norm_ex(data, roi, shape):
    # assert roi[3] < shape[0] and roi[4] < shape[1] and roi[5] < shape[2]
    data_shape = data.shape
    data_crop = numpy.zeros([data.shape[0], roi[3], roi[4], roi[5]], dtype=data.dtype)

    src_begin_0 = min(max(roi[0], 0), data.shape[-3])
    src_end_0 = min(max(roi[0] + roi[3], 0), data.shape[-3])
    src_begin_1 = min(max(roi[1], 0), data.shape[-2])
    src_end_1 = min(max(roi[1] + roi[4], 0), data.shape[-2])
    src_begin_2 = min(max(roi[2], 0), data.shape[-1])
    src_end_2 = min(max(roi[2] + roi[5], 0), data.shape[-1])
    
    dst_begin_0 = src_begin_0 - roi[0]
    dst_end_0 = dst_begin_0 + src_end_0 - src_begin_0
    dst_begin_1 = src_begin_1 - roi[1]
    dst_end_1 = dst_begin_1 + src_end_1 - src_begin_1    
    dst_begin_2 = src_begin_2 - roi[2]
    dst_end_2 = dst_begin_2 + src_end_2 - src_begin_2

    src_roi = [src_begin_0, src_begin_1, src_begin_2, src_end_0 - src_begin_0, src_end_1 - src_begin_1, src_end_2 - src_begin_2]
    dst_roi = [dst_begin_0, dst_begin_1, dst_begin_2, dst_end_0 - dst_begin_0, dst_end_1 - dst_begin_1, dst_end_2 - dst_begin_2]

    # print(src_roi, dst_roi)

    data_crop[:,
    dst_begin_0: dst_end_0,
    dst_begin_1: dst_end_1,
    dst_begin_2: dst_end_2
    ] = data[:,
        src_begin_0: src_end_0,
        src_begin_1: src_end_1,
        src_begin_2: src_end_2]


    return data_crop, src_roi, dst_roi



def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def CVS2Dict(filePath):
    file = csv.reader(open(filePath, "r"))
    
    allSample = [line for line in file]

    key = allSample[0]
    allSample = allSample[1:]
    return [dict(zip(key, sample)) for sample in allSample]
    
    
def CVS2Disk(filePath, data):

    with open(filePath, "w") as f:
        f_csv = csv.writer(f)
        for d in data:
            f_csv.writerow(d)


def OneHot(label, nClass):
    out = numpy.zeros((label.size, nClass), dtype=label.dtype)
    out[numpy.arange(label.size), label.reshape([-1]).astype(numpy.uint8)] = 1
    shape = list(label.shape) + [-1]
    out = out.reshape(shape)
    return out

def Dice(label, result, nClass):
    labelOneHot = OneHot(label, nClass)
    resultOneHot = OneHot(result, nClass)

    top = labelOneHot * resultOneHot
    top = numpy.sum(top, axis=0)
    top = numpy.sum(top, axis=0)
    top = numpy.sum(top, axis=0)

    labelOneHot = numpy.sum(labelOneHot, axis=0)
    labelOneHot = numpy.sum(labelOneHot, axis=0)
    labelOneHot = numpy.sum(labelOneHot, axis=0)

    resultOneHot = numpy.sum(resultOneHot, axis=0)
    resultOneHot = numpy.sum(resultOneHot, axis=0)
    resultOneHot = numpy.sum(resultOneHot, axis=0)
    bot = labelOneHot + resultOneHot

    dice = [t * 2 / b for t, b in zip(top, bot)]
    
    return dice

def ReadImage(fileName):
    if len(fileName) > 260:
        print("filename is too long")
        return None


    image = cv2.imdecode(numpy.fromfile(fileName, dtype=numpy.uint8), -1)
    if image is None:
        return None

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image

ImageCondition = ["jpg", "jpeg", "bmp", "tif", "png", "ppm", "pgm"]
def FindAllFile(filePath, blackList = [], fileCondition = ImageCondition, nSamples=None):
    allFileName = []
    for dirpath, dirnames, filenames in os.walk(filePath):
        #print('Directory', dirpath)
        for filename in filenames:
            suffix = filename.split(".")[-1].lower()
            if filename in blackList:
                continue
            if suffix in fileCondition:
                allFileName.append(os.path.join(dirpath, filename))

        if nSamples is not None and len(allFileName) > nSamples:
            break
    return allFileName

def FindAllFolder(filePath, blackList = [], fileCondition = [], nSamples=None):
    allFolderName = []
    for dirpath, dirnames, filenames in os.walk(filePath):
        for dirname in dirnames:
            if fileCondition is None or (dirname in fileCondition and (dirname not in blackList)):
                allFolderName.append(os.path.join(dirpath, dirname))
        if nSamples is not None and len(allFolderName) > nSamples:
            break
    return allFolderName


def CircleRandom(random, radius):
    x = 0.
    y = 0.
    r = radius * radius
    while True:
        x = random.random() * radius
        y = random.random() * radius
        if x * x + y * y > r:
            break
    return x, y


def PtsInRect(point, rect):
    if rect[0] < int(point[0]) < rect[0] + rect[2] and rect[1] < int(point[1]) < rect[1] + rect[3]:
        return True
    else:
        return False

def GetRectTLBR(rect):
    return (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3]))

def CutImage(image, tl, br):
    imageH = image.shape[0]
    imageW = image.shape[1]
    imageD = image.shape[2]

    roiW = br[0] - tl[0]
    roiH = br[1] - tl[1]


    imageXBegin = max(0, tl[0])
    imageXEnd = min(imageW, br[0])
    imageYBegin = max(0, tl[1])
    imageYEnd = min(imageH, br[1])


    cutXBegin = imageXBegin - tl[0]
    cutXEnd = cutXBegin + imageXEnd - imageXBegin
    cutYBegin = imageYBegin - tl[1]
    cutYEnd = cutYBegin + imageYEnd - imageYBegin



    imageCut = numpy.zeros([roiH, roiW, imageD], dtype=image.dtype)
    try:
        imageCut[cutYBegin: cutYEnd, cutXBegin:cutXEnd] = image[imageYBegin: imageYEnd, imageXBegin: imageXEnd]
    except ValueError:
        print(tl, br)
        print(imageXBegin, imageXEnd, imageYBegin, imageYEnd)
        print(cutXBegin, cutXEnd, cutYBegin, cutYEnd)
        print(image.shape)
        imageS = image[imageXBegin: imageXEnd, imageYBegin: imageYEnd]
        print(imageS.shape)
        cv2.imshow("imageCut", imageCut)
        cv2.imshow("image", image)
        cv2.rectangle(image, (int(imageXBegin), int(imageYBegin)), (int(imageXEnd), int(imageYEnd)),(255,0,0))
        cv2.waitKey(0)
    return imageCut

class CalReceptiveField:

    def __init__(self, allFunc):
        self.allFunc = allFunc

    def Add(self, netSize, layerFunc):
        netSize = netSize + layerFunc

    def Div(self, netSize, layerFunc):
        netSize = netSize / layerFunc

    def Sub(self, netSize, layerFunc):
        netSize = netSize - layerFunc

    def Mul(self, netSize, layerFunc):
        netSize = netSize * layerFunc

    def ConvB2T(self, netSize, kernelSize, stride):
        netSize = math.ceil((netSize - kernelSize + 1) / stride)

    def ConvT2B(self, netSize, kernelSize, stride):
        netSize = netSize * stride + kernelSize - 1

    def PoolB2T(self, netSize, stride):
        netSize = netSize // stride

    def PoolT2B(self, netSize, stride):
        netSize = netSize * stride


    # def CalReceptiveFieldOneLayer(self, layerIdx, outPutSize = 1):
    #     for i in self.allFunc[0, layerIdx, -1]:

