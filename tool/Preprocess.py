import os
import numpy
import tensorflow as tf
import random
import cv2
import tool.ToolBox as tBox
import tool.Face as tFace
import math

from tensorflow.python.ops import control_flow_ops

def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]

def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)

def ColorSample(image):

    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)


    image = apply_with_random_selector(
        image,
        lambda x, ordering: distort_color(x, ordering, False),
        num_cases=4)

    image = tf.multiply(tf.subtract(image, 0.5), 2.0)

    return image

def ResizeImage(image, height, width):
    return apply_with_random_selector(
        image,
        lambda x, method: tf.image.resize_images(x, [height, width], method=method),
        num_cases=4)

def MirrorSample(sample):
    if sample.pose == sample.PoseLeft or sample.pose == sample.PoseRight or random.uniform(0., 1.) > 0.5:
        return sample


    # imageRaw = sample.image

    sample.image = numpy.array(numpy.fliplr(sample.image))

    nCol = sample.image.shape[1]

    if sample.face is not None:
        sample.face[0] = nCol - sample.face[0] - sample.face[2]

    if sample.keyPoint is not None:
        sample.keyPoint = tFace.MirrorKeyPoint(sample.keyPoint, nCol)

    # cv2.imshow("a", imageRaw)
    # sample.Show()


    return sample

def RotateSample(sample):

    # if sample.keyPoint is None:
    #     nR = random.choice([0, 1, 2, 3])
    #     cx = sample.face[0] + sample.face[2] / 2
    #     cy = sample.face[1] + sample.face[3] / 2
    #     angle = nR * 90
    # else:
    #     dAngle = 180
    #
    #     image = sample.image
    #     keyPoint = sample.keyPoint
    #     keyPoint5p = tFace.PtsN25(keyPoint)
    #     r = cv2.minEnclosingCircle(keyPoint5p)
    #
    #     angle = random.uniform(-dAngle, dAngle)
    #
    #     cx = int(r[0][0])
    #     cy = int(r[0][1])
    #
    #     nR = (angle + dAngle) // 90


    dAngle = 180
    angle = random.uniform(-dAngle, dAngle)
    cx = sample.face[0] + sample.face[2] / 2
    cy = sample.face[1] + sample.face[3] / 2
    nR = (angle + dAngle) // 90



    rMat = cv2.getRotationMatrix2D((cx, cy), angle, 1)

    sample.rotateAngle = angle / 180
    sample.image = cv2.warpAffine(sample.image, rMat, sample.image.shape[0:2])

    if sample.keyPoint is not None:
        nPoint = len(sample.keyPoint)
        for p_n in range(nPoint):
            x = sample.keyPoint[p_n][0]
            y = sample.keyPoint[p_n][1]
            sample.keyPoint[p_n][0] = x * rMat[0][0] + y * rMat[0][1] + rMat[0][2]
            sample.keyPoint[p_n][1] = x * rMat[1][0] + y * rMat[1][1] + rMat[1][2]

    #     keyPoint5p = tFace.PtsN25(sample.keyPoint)
    #     sample.face = tFace.FaceRectBy5Pts(keyPoint5p)
    # else:
    x = sample.face[0]
    y = sample.face[1]
    xtl = x * rMat[0][0] + y * rMat[0][1] + rMat[0][2]
    ytl = x * rMat[1][0] + y * rMat[1][1] + rMat[1][2]

    x = sample.face[0] + sample.face[2]
    y = sample.face[1]
    xtr = x * rMat[0][0] + y * rMat[0][1] + rMat[0][2]
    ytr = x * rMat[1][0] + y * rMat[1][1] + rMat[1][2]

    x = sample.face[0]
    y = sample.face[1] + sample.face[3]
    xbl = x * rMat[0][0] + y * rMat[0][1] + rMat[0][2]
    ybl = x * rMat[1][0] + y * rMat[1][1] + rMat[1][2]

    x = sample.face[0] + sample.face[2]
    y = sample.face[1] + sample.face[3]
    xbr = x * rMat[0][0] + y * rMat[0][1] + rMat[0][2]
    ybr = x * rMat[1][0] + y * rMat[1][1] + rMat[1][2]

    xtled = min(xtl, xtr, xbl, xbr)
    ytled = min(ytl, ytr, ybl, ybr)
    xbred = max(xtl, xtr, xbl, xbr)
    ybred = max(ytl, ytr, ybl, ybr)

    oldArea = sample.face[2] * sample.face[3]
    newArea = (ybred - ytled) * (xbred - xtled)
    rate = max(oldArea / newArea, 0.8)
    cx = (xtled + xbred) * 0.5
    rx = (xbred - xtled) * 0.5 * rate
    cy = (ybred + ytled) * 0.5
    ry = (ybred - ytled) * 0.5 * rate

    sample.face[0] = cx - rx
    sample.face[1] = cy - ry
    sample.face[2] = rx * 2
    sample.face[3] = ry * 2

    # sample.face[0] = xtled
    # sample.face[1] = ytled
    # sample.face[2] = xbred - xtled
    # sample.face[3] = ybred - ytled




    if sample.faceLabel != 0:
        # sample.faceLabel = (sample.faceLabel - 1 + nR) % 4 + 1
        sample.faceLabel = 1

def BlurSample(sample):
    if random.uniform(0., 1.) > 0.8:
        return sample

    size = random.choice([3, 5, 7])
    sample.image = cv2.blur(sample.image, (size, size))

    return sample

def TestureSample(sample):
    if random.uniform(0., 1.) > 0.8:
        return sample

    for l in range(int(numpy.random.uniform(1, 15))):
        line = numpy.random.uniform(0, sample.image.shape[0], size=[2, 2])
        color = (random.uniform(0, 255), random.uniform(0, 255), random.uniform(0, 255))
        cv2.line(sample.image, (int(line[0][0]), int(line[0][1])), (int(line[1][0]), int(line[1][1])),
                 color=color, thickness=int(random.uniform(1, 2)))

    # sample.Show()
    return sample

def _DisturbRect(rect, disRate):
    dw = rect[2] * disRate
    dh = rect[3] * disRate
    x1 = rect[0] + random.uniform(-dw, dw)
    y1 = rect[1] + random.uniform(-dh, dh)
    x2 = rect[0] + rect[2] + random.uniform(-dw, dw)
    y2 = rect[1] + rect[3] + random.uniform(-dh, dh)
    return [x1, y1, x2 - x1, y2 - y1]

def _ExpandRect(rect, scale, pad):
    cx = rect[0] + rect[2] / 2
    cy = rect[1] + rect[3] / 2

    r = max(rect[2], rect[3]) / 2
    r2 = r * 2

    # add pad
    r2 = r2 * (scale + pad * 2) / scale
    r = r2 / 2

    return [cx - r, cy - r, r * 2, r * 2]

def _EncodeRect(faceRect, trainRect):
    x1 = (faceRect[0] - trainRect[0]) / trainRect[2]
    x2 = (faceRect[0] + faceRect[2] - trainRect[0] - trainRect[2]) / trainRect[2]
    y1 = (faceRect[1] - trainRect[1]) / trainRect[3]
    y2 = (faceRect[1] + faceRect[3] - trainRect[1] - trainRect[3]) / trainRect[3]
    return [x1, y1, x2, y2]

def _DecodeRect(trainRect, faceD):
    x1 = trainRect[0] + faceD[0] * trainRect[2]
    y1 = trainRect[1] + faceD[1] * trainRect[3]
    x2 = trainRect[0] + trainRect[2] + faceD[2] * trainRect[2]
    y2 = trainRect[1] + trainRect[3] + faceD[3] * trainRect[3]
    return [x1, y1, x2 - x1, y2 - y1]

def CalAngle(xb, yb, xc, yc, xe, ye):
    xbc = xb - xc
    ybc = yb - yc

    xec = xe - xc
    yec = ye - yc

    direct = xbc * yec - ybc * xec

    if direct > 0:
        direct = 1
    else:
        direct = -1

    cosValue = (xbc * xec + ybc * yec) / (math.sqrt(xbc * xbc + ybc * ybc) * math.sqrt(xec * xec + yec * yec))

    cosValue = min(max(-1., cosValue), 1.)
    radian = math.acos(cosValue) * direct
    angle = radian * 180 / math.pi

    return angle, radian


def Cart2Polar(xc, yc, x, y):
    angle, radian = CalAngle(x, y, xc, yc, xc, 0)

    xCen = x - xc
    yCen = y - yc
    rho = math.sqrt(xCen * xCen + yCen * yCen)

    return rho, radian


def Polar2Cart(xc, yc, rho, theta):
    xCen = -math.cos(theta) * rho
    yCen = -math.sin(theta) * rho


    return yc + yCen, xc + xCen



def CropSample(sample, scale, pad):
    if sample.faceLabel == 0:
        dis = 0
    else:
        dis = 0.2

    if sample.face is None and sample.keyPoint is None:
        return


    while True:
        faceRectDis = _DisturbRect(sample.face, dis)
        iou = tFace.CalIOU(sample.face, faceRectDis)
        if iou > 0.8:
            faceRectExp = _ExpandRect(faceRectDis, scale=scale, pad=pad)
            sample.faceD = _EncodeRect(faceRect=sample.face, trainRect=faceRectExp)

            # cv2.rectangle(sample.image,
            #               (int(faceRectExp[0]), int(faceRectExp[1])),
            #               (int(faceRectExp[0] + faceRectExp[2]), int(faceRectExp[1] + faceRectExp[3])), (255,0,255))
            #
            # cv2.rectangle(sample.image,
            #               (int(faceRectDis[0]), int(faceRectDis[1])),
            #               (int(faceRectDis[0] + faceRectDis[2]), int(faceRectDis[1] + faceRectDis[3])), (0,0,255))
            #
            # cv2.rectangle(sample.image,
            #               (int(sample.face[0]), int(sample.face[1])),
            #               (int(sample.face[0] + sample.face[2]), int(sample.face[1] + sample.face[3])), (255,255,255), 2)

            #
            # x1 = faceRectExp[0] + sample.faceD[0] * faceRectExp[2]
            # y1 = faceRectExp[1] + sample.faceD[1] * faceRectExp[3]
            # x2 = faceRectExp[0] + faceRectExp[2] + sample.faceD[2] * faceRectExp[2]
            # y2 = faceRectExp[1] + faceRectExp[3] + sample.faceD[3] * faceRectExp[3]
            # faceRect = _DecodeRect(faceRectExp, sample.faceD)
            # cv2.rectangle(sample.image,
            #               (int(faceRect[0]), int(faceRect[1])),
            #               (int(faceRect[0] + faceRect[2]), int(faceRect[1] + faceRect[3])), 255)
            #
            # cv2.imshow("a", sample.image)
            # cv2.waitKey(0)

            sample.image = tBox.CutImage(sample.image,
                                            (int(faceRectExp[0]), int(faceRectExp[1])),
                                            (int(faceRectExp[0] + faceRectExp[2]), int(faceRectExp[1] + faceRectExp[3])))

            if sample.keyPoint is not None:
                size = faceRectExp[2]


                sample.keyPoint = numpy.array(
                    [[p[0] - faceRectExp[0] - faceRectExp[2] * 0.5, p[1] - faceRectExp[1] - faceRectExp[3] * 0.5] for p in sample.keyPoint]) / size

                # cx = faceRectExp[2] * 0.5
                # cy = faceRectExp[3] * 0.5
                # for p_n in range(len(sample.keyPoint)):
                #     x = sample.keyPoint[p_n][0] - faceRectExp[0]
                #     y = sample.keyPoint[p_n][1] - faceRectExp[1]
                #     rho, theta = Cart2Polar(cx, cy, x, y)
                #
                #     sample.keyPoint[p_n][0] = rho / (size)
                #     sample.keyPoint[p_n][1] = theta / (math.pi)
            #
            sample.face = [0, 0, faceRectExp[2], faceRectExp[3]]
            # sample.Show()
            break

        # tl = [int(0 + sample.faceD[0]), int(0 + sample.faceD[1])]
        # br = [int(sample.image.shape[0] + sample.faceD[2]), int(sample.image.shape[1] + sample.faceD[3])]
        # cv2.rectangle(sample.image, (tl[0], tl[1]), (br[0], br[1]), (0,0,255))
        # cv2.rectangle(sample.image, (int(sample.face[0]), int(sample.face[1])), (int(sample.face[0] + sample.face[2]), int(sample.face[1] + sample.face[3])), (0, 255, 255))
        # cv2.imshow("111a", sample.image)
        # cv2.waitKey(0)
        # while True:
        #     w = sample.face[2]
        #     h = sample.face[3]
        #     rx = int(random.uniform(0, 1) + 0.5)
        #     cx = rx * sample.face[0] + (1 - rx) * (sample.face[0] + sample.face[2])
        #     ry = int(random.uniform(0, 1) + 0.5)
        #     cy = ry * sample.face[1] + (1 - ry) * (sample.face[1] + sample.face[3])
        #
        #     faceRect = _DisturbRect([cx - w / 2, cy - h / 2, w, h], dis)
        #     iou = tFace.CalIOU(sample.face, faceRect)
        #     if iou < 0.4:
        #         faceRectNeg, _ = _ExpandRect(faceRect)
        #         sample.imageNeg = tBox.CutImage(sample.image,
        #                                         (int(faceRectNeg[0]), int(faceRectNeg[1])),
        #                                         (int(faceRectNeg[0] + faceRectNeg[2]), int(faceRectNeg[1] + faceRectNeg[3])))
        #         break

        # cv2.rectangle(sample.image, (int(faceRectPos[0]), int(faceRectPos[1])),
        #               (int(faceRectPos[0] + faceRectPos[2]), int(faceRectPos[1] + faceRectPos[3])), (255))
        # cv2.rectangle(sample.image, (int(faceRectNeg[0]), int(faceRectNeg[1])),
        #               (int(faceRectNeg[0] + faceRectNeg[2]), int(faceRectNeg[1] + faceRectNeg[3])), (255))
        #
        # sample.Show()

    return sample

def CoveredSample(sample):
    if random.uniform(0., 1.) > 0.8:
        return sample

    h, w = sample.image.shape[0: 2]
    x = int(random.uniform(0, w))
    y = int(random.uniform(0, h))

    if x < w / 2:
        xBegin = 0
        xEnd = x
    else:
        xBegin = x
        xEnd = w

    if y < h / 2:
        yBegin = 0
        yEnd = y
    else:
        yBegin = y
        yEnd = h

    sample.image[yBegin: yEnd, xBegin: xEnd, :] = 0


def ResizeSample(sample, height, width):
    rate = sample.image.shape[0] / height
    sample.image = cv2.resize(sample.image, (height, width))
    sample.face = [v / rate for v in sample.face]

class FaceSample:
    PoseLeft = 0
    PoseFront = 1
    PoseRight = 2

    def __init__(self, nRow, nCol, nChannel, image, faceLabel, face, keyPoint, pose):

        nChannel = image.size // (nRow * nCol)
        self.image = image.reshape([nRow, nCol, nChannel])
        if nChannel != 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            nChannel = 3

        self.faceLabel = faceLabel

        self.face = face.copy()

        self.faceD = None

        self.keyPoint = keyPoint
        if self.keyPoint is not None:
            self.keyPoint = self.keyPoint.reshape([-1, 2])

        self.pose = pose


    def ToList(self):

        if self.keyPoint is None:

            return [
                self.image,
                numpy.array([self.faceLabel], dtype=numpy.int),
                numpy.array(self.faceD, dtype=numpy.float32),
            ]

        else:

            return [
                self.image,
                numpy.array([self.faceLabel], dtype=numpy.int),
                numpy.array(self.faceD, dtype=numpy.float32),

                numpy.array([self.keyPoint.reshape([-1])], dtype=numpy.float32),
                numpy.array([self.pose], dtype=numpy.int),
                numpy.array([self.rotateAngle], dtype=numpy.float32)
            ]



    def Show(self, image = None, nameScope = "imageShow", delay = 0):

        if image is None:
            imageShow = self.image.copy()
        else:
            imageShow = image.copy()

        if self.face is not None and self.faceD is not None:
            face = _DecodeRect(self.face, self.faceD)
            cv2.rectangle(imageShow,
                          (int(face[0]), int(face[1])),
                          (int(face[0]+face[2]), int(face[1]+face[3])), (255), 1)
        elif self.face is not None and self.faceD is None:
            cv2.rectangle(imageShow,
                          (int(self.face[0]), int(self.face[1])),
                          (int(self.face[0]+self.face[2]), int(self.face[1]+self.face[3])), (255))

        if self.keyPoint is not None:
            print(len(self.keyPoint))
            for p in self.keyPoint:
                cv2.circle(imageShow, (int(p[0] * self.face[2] + self.face[2] * 0.5), int(p[1] * self.face[3] + self.face[3] * 0.5)), 2, 255)

        if self.pose is not None:
            print(nameScope, "pose", self.pose)
        else:
            print(nameScope, "pose", "None")

        if self.faceLabel is not None:
            if type(self.faceLabel) == type(1):
                a = 0
            print("faceLabel", self.faceLabel)
        else:
            print("faceLabel", "None")



        cv2.imshow(nameScope, imageShow)

        if delay >= 0:
            cv2.waitKey(delay)


def DataPreprocess(nRow, nCol, nChannel, image, faceLabel, face, pad, scale, keyPoint = None, pose = None):
    # cv2.imshow("a", image)
    sample = FaceSample(nRow=nRow, nCol=nCol, nChannel=nChannel, image=image,
                        faceLabel=faceLabel, face=face, keyPoint=keyPoint, pose=pose)

    MirrorSample(sample)

    RotateSample(sample)

    TestureSample(sample)

    if scale > 64:
        BlurSample(sample)

    CropSample(sample, pad=pad, scale=scale)

    CoveredSample(sample)

    ResizeSample(sample, scale + pad * 2, scale + pad * 2)

    # sample.Show()
    return sample.ToList()



import FaceDetection as fd
import copy
if __name__ == '__main__':
    posSample = r"D:\zyx\Environment\data\active\Face\Nor\image\pos.192\3764_z_0_8690_143356687074465240.jpg"
    negSample = r"D:\zyx\Environment\data\active\Face\Nor\image\neg.192\0\1.jpg"
    pointSample = r"D:\zyx\Environment\data\active\Face\NorPoint\image\92.front\allSample_Test_set2_D000008.jpg"

    print("test pos")
    image = tBox.ReadImage(posSample)
    face = tFace.LoadFace(posSample, "")[0]
    nRow, nCol, nChannel = image.shape
    samplePos = FaceSample(nCol=nCol, nRow=nRow, nChannel=nChannel, image=image, face=face, faceLabel=1,
                              keyPoint=None, pose=None)





    image = tBox.ReadImage(negSample)
    face = tFace.LoadFace(negSample, "")[0]
    nRow, nCol, nChannel = image.shape
    sampleNeg = FaceSample(nCol=nCol, nRow=nRow, nChannel=nChannel, image=image, face=face, faceLabel=0,
                              keyPoint=None, pose=None)

    image = tBox.ReadImage(pointSample)
    if pointSample.find("left") != -1:
        pose = fd.DataFaceAlignment.poseLeft
    elif pointSample.find("front") != -1:
        pose = fd.DataFaceAlignment.poseFront
    elif pointSample.find("right") != -1:
        pose = fd.DataFaceAlignment.poseRight
    else:
        print("no pose info")

    face = tFace.LoadFace(pointSample, "")[0]
    keyPoint, _, _ = tFace.LoadPts(pointSample, "pts")
    nRow, nCol, nChannel = image.shape
    samplePoint = FaceSample(nCol=nCol, nRow=nRow, nChannel=nChannel, image=image, face=face, faceLabel=1,
                                keyPoint=keyPoint, pose=pose)

    # allPloar = []
    # allCartRaw = []
    # cx = image.shape[0] / 2
    # cy = image.shape[1] / 2
    # for p in keyPoint:
    #     rho, theta = Cart2Polar(cx, cy, p[0], p[1])
    #     allCartRaw.append([p[0] * 0.5, p[1] * 0.5])
    #     allPloar.append([rho / (cx * 2), theta / math.pi])
    #
    # maxDif = 0
    # allCart = []
    # for ploar, point in zip(allPloar, keyPoint):
    #     x, y = Polar2Cart(cx / 2, cy / 2, ploar[0] * (cx), ploar[1] * math.pi)
    #     allCart.append([x, y])
    #     v = (x - point[0]) * (x - point[0]) + (y - point[1]) * (y - point[1])
    #     if maxDif < v:
    #         maxDif = v









    allSample = [samplePos, sampleNeg, samplePoint]

    # for i in range(10):
    #     print("test MirrorSample..", i)
    #     for s in allSample:
    #         sOld = copy.deepcopy(s)
    #         MirrorSample(s)
    #         sOld.Show(nameScope="RawSample", delay=5)
    #         s.Show(nameScope="MirrorSample", delay=0)

    # for i in range(10):
    #     print("test RotateSample..", i)
    #     for s in allSample:
    #         sOld = copy.deepcopy(s)
    #         RotateSample(s)
    #         sOld.Show(nameScope="RawSample", delay=-1)
    #         s.Show(nameScope="NewSample", delay=0)

    for i in range(100):
        print("test RotateSample..", i)
        for s in allSample:
            sOld = copy.deepcopy(s)
            RotateSample(s)
            sOld.Show(nameScope="RawSample", delay=-1)
            s.Show(nameScope="NewSample", delay=0)

    # for i in range(10):
    #     print("test RotateSample..", i)
    #     for s in allSample:
    #         sOld = copy.deepcopy(s)
    #         TestureSample(s)
    #         sOld.Show(nameScope="RawSample", delay=-1)
    #         s.Show(nameScope="NewSample", delay=0)

    # for i in range(10):
    #     print("test RotateSample..", i)
    #     for s in allSample:
    #         sOld = copy.deepcopy(s)
    #
    #         CoveredSample(s)
    #
    #         sOld.Show(nameScope="RawSample", delay=-1)
    #         s.Show(nameScope="NewSample", delay=0)