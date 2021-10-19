import numpy
from scipy.ndimage.interpolation import zoom
import scipy.ndimage.interpolation
import xml
import cv2

import sys
sys.path.append("../")
import tool.SegToolBox as stBox
import random

import tool.MedicalImagePreprocess as mp

import torch
import nibabel as nib
nib.Nifti1Header.quaternion_threshold = - numpy.finfo(numpy.float32).eps * 10

import math
class RandomFlip:

    def __init__(self, label_xml):
        dom = xml.dom.minidom.parse(label_xml)
        root = dom.documentElement
        all_region = root.getElementsByTagName("Label")
        all_region_name = [(r.getElementsByTagName("Name")[0].firstChild.data, int(r.getElementsByTagName("Number")[0].firstChild.data)) for r in
                           all_region if
                           (int(r.getElementsByTagName("Number")[0].firstChild.data) not in stBox.LabelMapIgnore) and
                           (int(r.getElementsByTagName("Number")[0].firstChild.data) in stBox.LabelMap.keys())]

        self.all_region_left = {r[0]: r[1] for r in all_region_name if r[0].find("Left") != -1}
        self.all_region_right = {r[0]: r[1] for r in all_region_name if r[0].find("Right") != -1}
        self.all_region_mid = {r[0]: r[1] for r in all_region_name if (r[0].find("Left") == -1 and r[0].find("Right") == -1)}

        assert(len(self.all_region_right) == len(self.all_region_left))

        self.all_region_pair = []
        for key in self.all_region_left:
            key_flip = key.replace("Left", "Right")

            left_idx = self.all_region_left[key]
            right_idx = self.all_region_right[key_flip]
            self.all_region_pair.append([left_idx, right_idx])


    def __call__(self, sample):
        img = sample["data"]
        seg = sample["seg"]

        seg_flip = numpy.zeros_like(seg)

        for key in self.all_region_mid:
            seg_flip[seg == self.all_region_mid[key]] = self.all_region_mid[key]
        
        for p in self.all_region_pair:
            seg_flip[seg == p[0]] = p[1]
            seg_flip[seg == p[1]] = p[0]

        sample["data"] = img[::-1, ::, ::]
        sample["seg"] = seg_flip[::-1, ::, ::]

        return sample


class RandomBG:
    def __init__(self, data_path):
        self.all_file_name = [v for v in tBox.FindAllFile(data_path, [], ["gz"]) if v.find("_glm.nii.gz") == -1 and v.find("_pglm.nii.gz") == -1]

    def __call__(self, sample):

        if numpy.random.uniform(0., 1.) < 0.3:
            return sample

        file_name = self.all_file_name[int(numpy.random.uniform(0, len(self.all_file_name)))]
        img_u = nib.load(file_name).get_data()
        img_u = img_u.astype(numpy.float32) / numpy.max(img_u)

        seg_name = file_name.replace("images", "labels").replace(".nii.gz", "_glm.nii.gz")
        if os.path.exists(seg_name) is True:
            seg_u = nib.load(seg_name).get_data()
        else:
            seg_name = file_name.replace("images", "labels").replace(".nii.gz", "_pglm.nii.gz")
            seg_u = nib.load(seg_name).get_data()

        src = img_u
        src[seg_u != 0] = 0

        dst = numpy.zeros_like(sample["data"])
        dx = src.shape[0] - dst.shape[0]
        dy = src.shape[1] - dst.shape[1]
        dz = src.shape[2] - dst.shape[2]

        x_begin_src, x_begin_dst = (int(numpy.random.uniform(0, dx)), 0) if dx > 0 else (0, -int(numpy.random.uniform(0, dx)))
        x_end_src, x_end_dst = (x_begin_src + dst.shape[0], x_begin_dst + dst.shape[0]) if dx > 0 else (
        x_begin_src + src.shape[0], x_begin_dst + src.shape[0])

        y_begin_src, y_begin_dst = (int(numpy.random.uniform(0, dy)), 0) if dy > 0 else (0, -int(numpy.random.uniform(0, dy)))
        y_end_src, y_end_dst = (y_begin_src + dst.shape[1], y_begin_dst + dst.shape[1]) if dy > 0 else (
        y_begin_src + src.shape[1], y_begin_dst + src.shape[1])

        z_begin_src, z_begin_dst = (int(numpy.random.uniform(0, dz)), 0) if dz > 0 else (0, -int(numpy.random.uniform(0, dz)))
        z_end_src, z_end_dst = (z_begin_src + dst.shape[2], z_begin_dst + dst.shape[2]) if dz > 0 else (
        z_begin_src + src.shape[2], z_begin_dst + src.shape[2])

        dst[x_begin_dst: x_end_dst, y_begin_dst: y_end_dst, z_begin_dst: z_end_dst] = src[x_begin_src: x_end_src, y_begin_src: y_end_src,
                                                                                      z_begin_src: z_end_src]

        dst[sample["data"] != 0] = sample["data"][sample["data"] != 0]
        sample["data"] = dst
        return sample
        # for im in dst:
        #     cv2.imshow("a", im.astype(numpy.float) / im.max())
        #     cv2.waitKey()

class RandomRot:
    def __init__(self, head_size, brain_size, angle):
        self.head_size = head_size
        self.brain_size = brain_size
        self.angle = angle

    def __call__(self, sample):
        data = sample["data"]
        seg = sample["seg"]

        for i in range(3):
            angle = numpy.random.uniform(-self.angle, self.angle)
            axex = numpy.random.choice([v for v in range(len(data.shape))], 2, replace=False)
            data_ed = scipy.ndimage.rotate(data, angle, axes=axex, reshape=True, output=None, order=2, mode='constant', cval=0.0, prefilter=True)
            seg_ed = scipy.ndimage.rotate(seg, angle, axes=axex, reshape=True, output=None, order=0, mode='constant', cval=0.0, prefilter=True)

        _, rect_head = mp.MedicalImagePreprocess.ThresholdCrop(data_ed, 3)
        _, rect_brain = mp.MedicalImagePreprocess.ThresholdCrop(seg_ed, 3)

        if rect_head[3] > self.head_size[0] or rect_head[4] > self.head_size[1] or rect_head[5] > self.head_size[2]:
            sample["tr_flag"] = False
        elif rect_brain[3] > self.brain_size[0] or rect_brain[4] > self.brain_size[1] or rect_brain[5] > self.brain_size[2]:
            sample["tr_flag"] = False

        data_ed = data_ed[
                  rect_head[0]: rect_head[0] + rect_head[3],
                  rect_head[1]: rect_head[1] + rect_head[4],
                  rect_head[2]: rect_head[2] + rect_head[5],
                  ]

        seg_ed = seg_ed[
                  rect_head[0]: rect_head[0] + rect_head[3],
                  rect_head[1]: rect_head[1] + rect_head[4],
                  rect_head[2]: rect_head[2] + rect_head[5],
                  ]

        sample["data"] = data_ed
        sample["seg"] = seg_ed
        # for seed, imed in zip(seg_ed, data_ed):
        #     cv2.imshow("seed", seed.astype(numpy.float) / seed.max())
        #     cv2.imshow("imed", imed.astype(numpy.float) / imed.max())
        #     cv2.waitKey()
        return sample


class PadAndCrop:
    def __init__(self, nor_size, random = True):
        self.nor_size = nor_size
        self.random = random

    def __call__(self, sample):

        data = sample["data"]
        shape = data.shape



        dx = abs(shape[0] - self.nor_size[0])
        dx_0 = numpy.random.randint(0, dx) if dx > 0 else 0
        dx_0 = dx_0 if self.random is True else dx // 2

        dy = abs(shape[1] - self.nor_size[1])
        dy_0 = numpy.random.randint(0, dy) if dy > 0 else 0
        dy_0 = dy_0 if self.random is True else dy // 2

        dz = abs(shape[2] - self.nor_size[2])
        dz_0 = numpy.random.randint(0, dz) if dz > 0 else 0
        dz_0 = dz_0 if self.random is True else dz // 2

        x_cav_begin, x_data_begin = (0, dx_0) if shape[0] > self.nor_size[0] else (dx_0, 0)
        y_cav_begin, y_data_begin = (0, dy_0) if shape[1] > self.nor_size[1] else (dy_0, 0)
        z_cav_begin, z_data_begin = (0, dz_0) if shape[2] > self.nor_size[2] else (dz_0, 0)

        x_w = min(shape[0], self.nor_size[0])
        y_w = min(shape[1], self.nor_size[1])
        z_w = min(shape[2], self.nor_size[2])



        cav = numpy.zeros(self.nor_size, dtype=data.dtype)
        cav[
            x_cav_begin: x_cav_begin + x_w,
            y_cav_begin: y_cav_begin + y_w,
            z_cav_begin: z_cav_begin + z_w
        ] = data[
            x_data_begin: x_data_begin + x_w,
            y_data_begin: y_data_begin + y_w,
            z_data_begin: z_data_begin + z_w
        ]


        sample["data"] = cav



        if "seg_global" in sample:
            cav_seg = numpy.zeros(self.nor_size, dtype=sample["seg"].dtype)
            cav_seg[
            x_cav_begin: x_cav_begin + x_w,
            y_cav_begin: y_cav_begin + y_w,
            z_cav_begin: z_cav_begin + z_w
            ] = sample["seg_global"][
                x_data_begin: x_data_begin + x_w,
                y_data_begin: y_data_begin + y_w,
                z_data_begin: z_data_begin + z_w
                ]
            sample["seg_global"] = cav_seg



        if "seg" in sample:
            cav_seg = numpy.zeros(self.nor_size, dtype=sample["seg"].dtype)
            cav_seg[
            x_cav_begin: x_cav_begin + x_w,
            y_cav_begin: y_cav_begin + y_w,
            z_cav_begin: z_cav_begin + z_w
            ] = sample["seg"][
                x_data_begin: x_data_begin + x_w,
                y_data_begin: y_data_begin + y_w,
                z_data_begin: z_data_begin + z_w
                ]
            sample["seg"] = cav_seg

            # all_v = []
            # for i in range(135):
            #     v = numpy.sum(brain == i)
            #     all_v.append(v)
            # for im in brain:
            #     cv2.imshow("a", im.astype(numpy.float32) / numpy.max(im))
            #     cv2.waitKey()

        return sample

class Subsample_global:

    def __init__(self, zoom_rate):
        self.zoom_rate = zoom_rate

    def __call__(self, sample):

        # sample["data_global"] = scipy.ndimage.interpolation.zoom(sample["data"], self.zoom_rate, order=2)
        sample["seg_global"] = scipy.ndimage.interpolation.zoom(sample["seg"], self.zoom_rate, order=0)
        sample["roi_global"] = sample["roi"] * self.zoom_rate
        # for im in sample["data_skull"]:
        #     cv2.imshow("a", im.astype(numpy.float) / numpy.max(im))
        #     cv2.waitKey()

        return sample

class Hemisphere:

    whole_2_half = 0
    half_2_whole = 1

    def __init__(self, label_xml, rect, rect_sub, flag):

        dom = xml.dom.minidom.parse(label_xml)
        root = dom.documentElement
        all_region = root.getElementsByTagName("Label")
        all_region_name = [(r.getElementsByTagName("Name")[0].firstChild.data, int(r.getElementsByTagName("Number")[0].firstChild.data)) for r in
                           all_region if
                           (int(r.getElementsByTagName("Number")[0].firstChild.data) not in stBox.LabelMapIgnore) and
                           (int(r.getElementsByTagName("Number")[0].firstChild.data) in stBox.LabelMap.keys())]

        self.all_region_mid = [r for r in all_region_name if (r[0].find("Left") == -1 and r[0].find("Right") == -1)]
        self.all_region_left = [r for r in all_region_name if r[0].find("Left") != -1]
        self.all_region_right = [r for r in all_region_name if r[0].find("Right") != -1]

        self.rect = rect
        self.rect_sub = rect_sub
        self.flag = flag


    def whole_2_half(self, sample):

        seg = sample["seg"]

        seg_left = numpy.zeros_like(seg)
        seg_right = numpy.zeros_like(seg)

        for r_n in range(len(self.all_region_mid)):
            region = self.all_region_mid[r_n]
            seg_left[seg == region[1]] = r_n + 1
            seg_right[seg == region[1]] = r_n + 1

        num_mid = len(self.all_region_mid) + 1
        for r_n in range(len(self.all_region_left)):
            region = self.all_region_left[r_n]
            seg_left[seg == region[1]] = num_mid + r_n

            region = self.all_region_right[r_n]
            seg_right[seg == region[1]] = num_mid + r_n

        sample["seg_left"] = seg_left
        sample["seg_right"] = seg_right

        return sample

    def half_2_whole(self, sample):

        seg_left = sample["seg_left"]
        seg_right = sample["seg_right"]

        seg = numpy.zeros(self.rect, dtype=seg_left.dtype)

        for r_n in range(len(self.all_region_mid)):
            region = self.all_region_mid[r_n]
            seg_left[seg == r_n + 1] = region[1]
            seg_right[seg == r_n + 1] = region[1]

        num_mid = len(self.all_region_mid) + 1
        for r_n in range(len(self.all_region_left)):
            region = self.all_region_left[r_n]
            seg_left[seg == num_mid + r_n] = region[1]

            region = self.all_region_right[r_n]
            seg_right[seg == num_mid + r_n] = region[1]

        sample["seg_left"] = seg_left
        sample["seg_right"] = seg_right

        return sample


    def __call__(self, sample):

        if self.flag == self.whole_2_half:
            return self.whole_2_half(sample)
        else:
            return self.half_2_whole(sample)


class NonlinearTransformation:

    def bernstein_poly(self, i, n, t):
        """
         The Bernstein polynomial of n, i as a function of t
        """

        return scipy.special.comb(n, i) * (t ** (n - i)) * (1 - t) ** i

    def bezier_curve(self, points, nTimes=1000):
        """
           Given a set of control points, return the
           bezier curve defined by the control points.
           Control points should be a list of lists, or list of tuples
           such as [ [1,1],
                     [2,3],
                     [4,5], ..[Xn, Yn] ]
            nTimes is the number of time steps, defaults to 1000
            See http://processingjs.nihongoresources.com/bezierinfo/
        """

        nPoints = len(points)
        xPoints = numpy.array([p[0] for p in points])
        yPoints = numpy.array([p[1] for p in points])

        t = numpy.linspace(0.0, 1.0, nTimes)

        polynomial_array = numpy.array([self.bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

        xvals = numpy.dot(xPoints, polynomial_array)
        yvals = numpy.dot(yPoints, polynomial_array)

        return xvals, yvals

    def __call__(self, sample):

        image = sample["data"]



        points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
        # print(points)
        xpoints = [p[0] for p in points]
        ypoints = [p[1] for p in points]
        xvals, yvals = self.bezier_curve(points, nTimes=100000)
        if random.random() < -0.5:
            # Half change to get flip
            xvals = numpy.sort(xvals)
        else:
            xvals, yvals = numpy.sort(xvals), numpy.sort(yvals)
        nonlinear_x = numpy.interp(image, xvals, yvals)

        sample["data"] = nonlinear_x.astype(numpy.float32)


        # for im1, im2 in zip(image, nonlinear_x):
        #     v = numpy.sum(im1 - im2)
        #     cv2.imshow("a", im1.astype(numpy.float32) / numpy.max(im1))
        #     cv2.imshow("b", im2.astype(numpy.float32) / numpy.max(im2))
        #     cv2.waitKey()

        return sample

class SelfFlip:

    weight = [
        0.020014544896662118, 0.009066454744630172, 0.034401757582453917, 0.030896249431724622, 0.016763909500703694,
        0.016896539182590212, 0.0009044408492800643, 0.0049327812857809655, 0.005087285877792676, 0.000317283401320578,
        0.0003165673380718094, 0.0011683721667803658, 0.0011377659236906606, 7.71386357439265e-05, 7.794479818366083e-05,
        0.015893775856453053, 0.004429829577972947, 0.004690986393243183, 0.03657865006524995, 0.04214918942497871,
        0.001802218011070496, 0.0015177572842658311, 0.010737655504015894, 0.010763434535831637, 0.003689159445769754,
        0.0035668499088872823, 0.002108049495149544, 0.0020235282144978905, 0.0034484575506291662, 0.003292982968683401,
        0.19801308342806923, 0.003378685736091938, 0.007185675375278692, 0.005656753789563621, 0.032704466279521845,
        0.03353930703258828, 0.0035766246594094845, 0.003015262531947413, 0.0036566351964813356, 0.0036058523382674753,
        0.007845821383024117, 0.008926449303092276, 0.0014295372749962962, 0.0015849254514749465, 0.004366398565474055,
        0.004376550407973928, 0.003701599020865944, 0.003687167609098549, 0.0029913048184643486, 0.0031136872641309996,
        0.008691699182306807, 0.008827964247349286, 0.007595597701141373, 0.007534887490508093, 0.00405139762594612,
        0.004288601386295717, 0.0019789898792255756, 0.002001543162175441, 0.006458534177683546, 0.006043805561591493,
        0.0022100223349563806, 0.0022948843601424402, 0.0012832174353158263, 0.0012866245729688266, 0.0018559053175278898,
        0.001995510935421534, 0.006181396091521632, 0.005878711617629457, 0.0030975770585102125, 0.0029832365961510107,
        0.007378967803895534, 0.007358627843527956, 0.0007975594196578576, 0.0007776065743572253, 0.002716686196765054,
        0.002397718684465238, 0.0035241086649444556, 0.003397291716255406, 0.014560363086266691, 0.013316406798649857,
        0.005522655559242591, 0.005445690281207377, 0.00186614261526277, 0.002090000304075518, 0.0010300628333637103,
        0.0010620741136633145, 0.0051243046578687505, 0.00440560506378154, 0.003461669252903768, 0.0034390073981874322,
        0.00424870366809171, 0.004334271606146383, 0.009390748556109645, 0.00951828607114165, 0.003719839157916542,
        0.0034097967699192023, 0.001396431106426774, 0.001359163556990605, 0.005002869017561829, 0.004677722776236944,
        0.006696541799569606, 0.006836541905172492, 0.006383630849847322, 0.006072551366878492, 0.0014034219045207519,
        0.0012592638450899427, 0.005944701710083033, 0.005685888601708295, 0.007268558624864213, 0.0066992152832008635,
        0.0011429560151866665, 0.0011513449298627798, 0.00739718380582968, 0.007109735436259513, 0.012655565010615925,
        0.012460892927260742, 0.0010057949514707476, 0.0010246606698297426, 0.0028042170459262367, 0.0027025101599384006,
        0.0017032864824703987, 0.0015946857244227464, 0.003344686068132472, 0.003966118580018053, 0.0014102967650901045,
        0.0013908851199696231, 0.0019612981400699064, 0.0019680940084836947, 0.001903792972442335, 0.0018808142055314232,
        0.00408606735669816, 0.0038981230117855177, 0.009731033658873536, 0.009077805855722109]

    labels_135 = [0, 4, 11, 23, 30, 31, 32, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55, 56, 57, 58, 59, 60, 61, 62, 69, 71, 72, 73, 75,
                  76, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 128, 129,
                  132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 160,
                  161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
                  190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207]

    labels_133 = [0, 4, 11, 23, 30, 31, 32, 35, 36, 37, 38, 39, 40, 41, 44, 45, 47, 48, 49, 50, 51, 52, 55, 56, 57, 58, 59, 60, 61, 62, 71, 72, 73, 75,
                  76, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 128, 129,
                  132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 160,
                  161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
                  190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207]

    def __init__(self, crop_size, brain_size, label_xml="1103_3_glm_LabelMap.xml", num_label = 135):
        self.crop_size = crop_size
        self.brain_size = brain_size

        dom = xml.dom.minidom.parse(label_xml)
        root = dom.documentElement
        all_region = root.getElementsByTagName("Label")
        all_region_num_name = {int(r.getElementsByTagName("Number")[0].firstChild.data): r.getElementsByTagName("Name")[0].firstChild.data.lower() for r in all_region}
        all_region_num_name[0] = "bg"

        all_region_name_num = {r.getElementsByTagName("Name")[0].firstChild.data.lower(): int(r.getElementsByTagName("Number")[0].firstChild.data) for r in all_region}

        self.all_pair_b0 = {}

        # for i in range(10):
        #     v = numpy.random.choice([1 + i for i in range(len(self.weight))], 1, replace=False, p=[v for v in self.weight])
        #     print(v)



        labels = self.labels_133 if num_label == 133 else self.labels_135
        for k_n, key in enumerate(self.labels_133):
            name = all_region_num_name[key].lower()
            self.all_pair_b0[name] = {"id": k_n}


        for name in self.all_pair_b0:

            if name.find("left") != -1:
                mirror_name = name.replace("left", "right")
            elif name.find("right") != -1:
                mirror_name = name.replace("right", "left")
            else:
                mirror_name = name

            self.all_pair_b0[name]["m_id"] = self.all_pair_b0[mirror_name]["id"]

        self.index_mirror = [self.all_pair_b0[key]["m_id"] for key in self.all_pair_b0]

        self.trans = NonlinearTransformation()
        pass

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

        return  all_idx[all_dif.index(max(all_dif))]

    def __call__(self, sample):
        def crop_test(r1, r2):
            d1 = sample["data"][0][r1[0]: r1[0] + r1[3], r1[1]: r1[1] + r1[4], r1[2]: r1[2] + r1[5]]
            d2 = sample["data"][1][r2[0]: r2[0] + r2[3], r2[1]: r2[1] + r2[4], r2[2]: r2[2] + r2[5]]
            v = numpy.sum(d1 - d2[::-1, : ,:])
            return v

        img = sample["data"]
        img_0 = self.trans({"data": img})["data"]
        img_1 = self.trans({"data": img})["data"]
        sample["data"] = numpy.concatenate([img_0[None, ::, ::, ::], img_1[None, ::-1, ::, ::]], axis=0)

        shape = img.shape

        if "seg" not in sample:
            b_x_c = max(0, shape[0] // 2 - self.brain_size[0] // 2)
            b_y_c = max(0, shape[1] // 2 - self.brain_size[1] // 2)
            b_z_c = max(0, shape[2] // 2 - self.brain_size[2] // 2)
            b_x = int(numpy.random.uniform(b_x_c // 2, b_x_c))
            b_y = int(numpy.random.uniform(b_y_c // 4, b_y_c // 2))
            b_z = int(numpy.random.uniform(b_z_c // 2, b_z_c))
            b_x_w = min(b_x + self.brain_size[0], shape[0]) - b_x
            b_y_w = min(b_y + self.brain_size[1], shape[1]) - b_y
            b_z_w = min(b_z + self.brain_size[2], shape[2]) - b_z

            roi_global = [
            int(numpy.random.uniform(b_x, b_x + b_x_w - self.crop_size[0])),
            int(numpy.random.uniform(b_y, b_y + b_y_w - self.crop_size[1])),
            int(numpy.random.uniform(b_z, b_z + b_z_w - self.crop_size[2])),
            self.crop_size[0],
            self.crop_size[1],
            self.crop_size[2]
        ]
        else:
            seg = sample["seg"]
            shape = seg.shape
            x = [numpy.sum(seg[v, :, :]) for v in range(shape[0])]
            x_begin, x_end = self.split_idx(x)

            y = [numpy.sum(seg[:, v, :]) for v in range(shape[1])]
            y_begin, y_end = self.split_idx(y)

            z = [numpy.sum(seg[:, :, v]) for v in range(shape[2])]
            z_begin, z_end = self.split_idx(z)

            # seg_crop = seg[x_begin: x_end, y_begin: y_end, z_begin: z_end]
            # for se in seg_crop:
            #     cv2.imshow("a", se.astype(numpy.float) / numpy.max(se))
            #     cv2.waitKey()

            b_x_c = (x_begin + x_end) // 2
            b_y_c = (y_begin + y_end) // 2
            b_z_c = (z_begin + z_end) // 2
            b_x = min(max(0, b_x_c - self.brain_size[0] // 2 + int(numpy.random.uniform(-(self.brain_size[0] // 5), self.brain_size[0] // 5 + 0.01))), shape[0] - self.brain_size[0])
            b_y = min(max(0, b_y_c - self.brain_size[1] // 2 + int(numpy.random.uniform(-(self.brain_size[1] // 5), self.brain_size[1] // 5 + 0.01))), shape[1] - self.brain_size[1])
            b_z = min(max(0, b_z_c - self.brain_size[2] // 2 + int(numpy.random.uniform(-(self.brain_size[2] // 5), self.brain_size[2] // 5 + 0.01))), shape[2] - self.brain_size[2])
            b_x_w = self.brain_size[0]
            b_y_w = self.brain_size[1]
            b_z_w = self.brain_size[2]



            while True:
                idx = numpy.random.choice([1 + i for i in range(len(self.weight))], 1, replace=False, p=[v for v in self.weight])[0]
                cond_loc = numpy.nonzero(seg == idx)
                num_loc = cond_loc[0].shape[0]
                if num_loc > 0:
                    break
            loc_idx = int(numpy.random.uniform(0, num_loc))
            x_c = cond_loc[0][loc_idx]
            y_c = cond_loc[1][loc_idx]
            z_c = cond_loc[2][loc_idx]

            roi_global = [
                min(max(b_x, x_c - self.crop_size[0] // 2), b_x + b_x_w - self.crop_size[0]),
                min(max(b_y, y_c - self.crop_size[1] // 2), b_y + b_y_w - self.crop_size[1]),
                min(max(b_z, z_c - self.crop_size[2] // 2), b_z + b_z_w - self.crop_size[2]),
                self.crop_size[0],
                self.crop_size[1],
                self.crop_size[2]
            ]
            sample.pop("seg")
            # sample["loc_idx"] = loc_idx

        roi_brain = [b_x, b_y, b_z, b_x_w, b_y_w, b_z_w]
        roi_brain_mirror = [shape[0] - b_x - b_x_w, b_y, b_z, b_x_w, b_y_w, b_z_w]

        roi_global_mirror = [
            shape[0] - roi_global[0] - roi_global[3],
            roi_global[1],
            roi_global[2],
            self.crop_size[0],
            self.crop_size[1],
            self.crop_size[2]
        ]


        roi_local = [
            roi_global[0] - roi_brain[0],
            roi_global[1] - roi_brain[1],
            roi_global[2] - roi_brain[2],
            self.crop_size[0],
            self.crop_size[1],
            self.crop_size[2]
        ]

        roi_local_mirror = [
            roi_global_mirror[0] - roi_brain_mirror[0],
            roi_global_mirror[1] - roi_brain_mirror[1],
            roi_global_mirror[2] - roi_brain_mirror[2],
            self.crop_size[0],
            self.crop_size[1],
            self.crop_size[2]
        ]

        # v1 = crop_test(roi_brain_mirror, roi_brain)
        # v2 = crop_test(roi_global, roi_global_mirror)
        #
        # d1 = sample["data"][0][
        #      roi_brain[0]: roi_brain[0] + roi_brain[3],
        #      roi_brain[1]: roi_brain[1] + roi_brain[4],
        #      roi_brain[2]: roi_brain[2] + roi_brain[5]]
        # d2 = sample["data"][1][
        #      roi_brain_mirror[0]: roi_brain_mirror[0] + roi_brain_mirror[3],
        #      roi_brain_mirror[1]: roi_brain_mirror[1] + roi_brain_mirror[4],
        #      roi_brain_mirror[2]: roi_brain_mirror[2] + roi_brain_mirror[5]]
        # d1s = d1[roi_local[0]: roi_local[0] + roi_local[3], roi_local[1]: roi_local[1] + roi_local[4], roi_local[2]: roi_local[2] + roi_local[5]]
        # d2s = d2[roi_local_mirror[0]: roi_local_mirror[0] + roi_local_mirror[3], roi_local_mirror[1]: roi_local_mirror[1] + roi_local_mirror[4], roi_local_mirror[2]: roi_local_mirror[2] + roi_local_mirror[5]]
        # v3 = numpy.sum(d1s - d2s[::-1, :, :])
        #
        #
        #
        #
        # kernel_size_head_local = 5
        # opt_stride = 4
        #
        # arr = [i for i in range(self.crop_size[2] - kernel_size_head_local + 1)]
        # arr_inv = arr[::-1]
        # d1t = numpy.array(arr)
        # d11 = d1t[::opt_stride]
        #
        # d2t = numpy.array(arr_inv)
        # d22 = d2t[::opt_stride][::-1]
        #
        # d1o = d1[
        #            kernel_size_head_local // 2: -(kernel_size_head_local // 2): opt_stride,
        #            kernel_size_head_local // 2: -(kernel_size_head_local // 2): opt_stride,
        #            kernel_size_head_local // 2: -(kernel_size_head_local // 2): opt_stride]
        #
        # d2o = d2[
        #            kernel_size_head_local // 2: -(kernel_size_head_local // 2): opt_stride,
        #            kernel_size_head_local // 2: -(kernel_size_head_local // 2): opt_stride,
        #            kernel_size_head_local // 2: -(kernel_size_head_local // 2): opt_stride]
        # v4 = numpy.sum(d1o - d2o[::-1, :, :])
        # print(v1, v2)



        sample["roi_brain"] = numpy.array([roi_brain, roi_brain_mirror])
        sample["roi"] = numpy.array([roi_local, roi_local_mirror])



        return sample

    def mirror_result(self, result, dim):
        result_mirror = result.flip(dims=[dim])



        # all_result = [result_mirror[:, i: i+1] for i in range(result.size(1))]
        # all_result_mirror = []
        # for key in self.all_pair_b0:
        #     id = self.all_pair_b0[key]["id"]
        #     m_id = self.all_pair_b0[key]["m_id"]
        #     all_result_mirror.append(all_result[m_id])
        # result_mirror_1 = torch.cat(all_result_mirror, dim=1)

        index_mirror = torch.tensor(self.index_mirror, device=result.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        index_mirror = index_mirror.repeat([result.size(0), 1, result.size(2), result.size(3), result.size(4)])
        result_mirror = result_mirror.gather(dim=1, index=index_mirror)

        # v = torch.sum(result_mirror_1 - result_mirror_2)
        return result_mirror

    def test(self, opt_stride=3, kernel_size_head_local=5):

        def label2oneHot(label, num_class):
            shape = list(label.shape)
            shape_onehot = shape
            shape_onehot[1] = num_class

            label_onehot = torch.zeros(shape, dtype=label.dtype, device=label.device)
            label_onehot = label_onehot.scatter_(1, label.long(), 1)
            return label_onehot

        # file_raw = "/lustre/home/yxzhao/Work/Data/BrainRegion/Train/Raw/labels/training-labels/1000_3_glm.nii.gz"
        # file_mir = "/lustre/home/yxzhao/Work/Data/BrainRegion/Train/Raw/labels/training-labels-flip/1000_3_glm.nii.gz"

        file_raw_data = "/lustre/home/yxzhao/Work/Data/BrainRegion/Train/Raw/images/training-images/1000_3.nii.gz"
        file_raw_seg = "/lustre/home/yxzhao/Work/Data/BrainRegion/Train/Raw/labels/training-labels/1000_3_glm.nii.gz"
        file_mir = "/lustre/home/yxzhao/Work/Data/BrainRegion/Train/Raw/images/training-images-flip/1000_3.nii.gz"

        data_raw = nib.load(file_raw_data).get_data()
        seg_raw = nib.load(file_raw_seg).get_data()

        sample = self.__call__({"data": data_raw, "seg": seg_raw})


        for i in range(100):
            file_raw = "/lustre/home/yxzhao/Work/Data/BrainRegion/Train/Raw/labels/training-labels/1000_3_glm.nii.gz"

            file_mir = "/lustre/home/yxzhao/Work/Data/BrainRegion/Train/Raw/labels/training-labels-flip/1000_3_glm.nii.gz"

            data_raw = nib.load(file_raw).get_data()
            seg_raw = nib.load(file_raw).get_data()


            data_mir = nib.load(file_mir).get_data()




            # for _ in range(10):
            #     sample = self.__call__({"data": data_raw})
            #     roi_brain = sample["roi_brain"][0]
            #     roi_brain_mirror = sample["roi_brain"][1]
            #     a = 0
            sample = self.__call__({"data": data_raw})
            roi_brain = sample["roi_brain"][0]
            roi_brain_mirror = sample["roi_brain"][1]

            roi_raw = sample["roi"][0]
            roi_mir = sample["roi"][1]

            data_raw = label2oneHot(torch.tensor(data_raw).unsqueeze(0).unsqueeze(0), num_class=135)
            data_mir = label2oneHot(torch.tensor(data_mir).unsqueeze(0).unsqueeze(0), num_class=135)
            data_mir_mir = self.mirror_result(data_mir, dim=2)

            dd = torch.sum(data_raw - data_mir)
            da = torch.sum(torch.abs(data_raw - data_mir))
            ddf = torch.sum(data_raw - data_mir_mir)
            daf = torch.sum(torch.abs(data_raw - data_mir_mir))

            data_raw = data_raw[:, :,
                       roi_brain[0]: roi_brain[0] + roi_brain[3],
                       roi_brain[1]: roi_brain[1] + roi_brain[4],
                       roi_brain[2]: roi_brain[2] + roi_brain[5]
                       ][:, :,
                roi_raw[0]: roi_raw[0] + roi_raw[3],
                roi_raw[1]: roi_raw[1] + roi_raw[4],
                roi_raw[2]: roi_raw[2] + roi_raw[5],
                       ]
            data_mir = data_mir[:, :,
                       roi_brain_mirror[0]: roi_brain_mirror[0] + roi_brain_mirror[3],
                       roi_brain_mirror[1]: roi_brain_mirror[1] + roi_brain_mirror[4],
                       roi_brain_mirror[2]: roi_brain_mirror[2] + roi_brain_mirror[5]
                       ][:, :,
                    roi_mir[0]: roi_mir[0] + roi_mir[3],
                    roi_mir[1]: roi_mir[1] + roi_mir[4],
                    roi_mir[2]: roi_mir[2] + roi_mir[5],
                       ]
            data_mir_mir = self.mirror_result(data_mir, dim=2)

            dd = torch.sum(data_raw - data_mir)
            da = torch.sum(torch.abs(data_raw - data_mir))
            ddf = torch.sum(data_raw - data_mir_mir)
            daf = torch.sum(torch.abs(data_raw - data_mir_mir))

            margin=1


            data_raw = data_raw[:, :,
                                   kernel_size_head_local // 2: -(kernel_size_head_local // 2): opt_stride,
                                   kernel_size_head_local // 2: -(kernel_size_head_local // 2): opt_stride,
                                   kernel_size_head_local // 2: -(kernel_size_head_local // 2): opt_stride]

            data_mir = data_mir[:, :,
                                   kernel_size_head_local // 2: -(kernel_size_head_local // 2): opt_stride,
                                   kernel_size_head_local // 2: -(kernel_size_head_local // 2): opt_stride,
                                   kernel_size_head_local // 2: -(kernel_size_head_local // 2): opt_stride]

            data_mir_mir = self.mirror_result(data_mir, dim=2)

            dd = torch.sum(data_raw - data_mir)
            da = torch.sum(torch.abs(data_raw - data_mir))
            ddf = torch.sum(data_raw - data_mir_mir)
            daf = torch.sum(torch.abs(data_raw - data_mir_mir))

            print(i, roi_raw)
            pass

class SelfFlipEx(SelfFlip):

    def __init__(self, kernel_size, kernel_stride, crop_size, brain_size, label_xml="1103_3_glm_LabelMap.xml", num_label=135):
        super(SelfFlipEx, self).__init__(crop_size, brain_size, label_xml="1103_3_glm_LabelMap.xml", num_label=num_label)

        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride

    def __call__(self, sample):
        def crop_test(r1, r2):
            d1 = sample["data"][0][r1[0]: r1[0] + r1[3], r1[1]: r1[1] + r1[4], r1[2]: r1[2] + r1[5]]
            d2 = sample["data"][1][r2[0]: r2[0] + r2[3], r2[1]: r2[1] + r2[4], r2[2]: r2[2] + r2[5]]
            v = numpy.sum(d1 - d2[::-1, : ,:])
            return v

        img = sample["data"]
        img_0 = self.trans({"data": img})["data"]
        img_1 = self.trans({"data": img})["data"]

        # img_0 = img
        # img_1 = img
        sample["data"] = numpy.concatenate([img_0[None, ::, ::, ::], img_1[None, ::-1, ::, ::]], axis=0)

        shape = img.shape

        seg = sample["seg"]
        shape = seg.shape
        x = [numpy.sum(seg[v, :, :]) for v in range(shape[0])]
        x_begin, x_end = self.split_idx(x)

        y = [numpy.sum(seg[:, v, :]) for v in range(shape[1])]
        y_begin, y_end = self.split_idx(y)

        z = [numpy.sum(seg[:, :, v]) for v in range(shape[2])]
        z_begin, z_end = self.split_idx(z)

        #the last item is padding
        r_w_x = math.ceil((x_end - x_begin - self.kernel_size) / self.kernel_stride) * self.kernel_stride + self.kernel_size + self.kernel_size // 2 * 2
        r_w_y = math.ceil((y_end - y_begin - self.kernel_size) / self.kernel_stride) * self.kernel_stride + self.kernel_size + self.kernel_size // 2 * 2
        r_w_z = math.ceil((z_end - z_begin - self.kernel_size) / self.kernel_stride) * self.kernel_stride + self.kernel_size + self.kernel_size // 2 * 2

        if r_w_x > self.brain_size[0] or r_w_y > self.brain_size[1] or r_w_z > self.brain_size[2]:
            sample["tr_flag"] = False
            return sample

        dif_x = self.brain_size[0] - r_w_x
        b_x_0 = x_begin - (numpy.random.randint(0, dif_x) if dif_x != 0 else 0)
        b_x_0 = min(max(0, b_x_0), shape[0] - self.brain_size[0])
        b_x_1 = x_begin - (numpy.random.randint(0, dif_x) if dif_x != 0 else 0)
        b_x_1 = min(max(0, b_x_1), shape[0] - self.brain_size[0])

        dif_y = self.brain_size[1] - r_w_y
        b_y_0 = y_begin - (numpy.random.randint(0, dif_y) if dif_y != 0 else 0)
        b_y_0 = min(max(0, b_y_0), shape[1] - self.brain_size[1])
        b_y_1 = y_begin - (numpy.random.randint(0, dif_y) if dif_y != 0 else 0)
        b_y_1 = min(max(0, b_y_1), shape[1] - self.brain_size[1])

        dif_z = self.brain_size[2] - r_w_z
        b_z_0 = z_begin - (numpy.random.randint(0, dif_z) if dif_z != 0 else 0)
        b_z_0 = min(max(0, b_z_0), shape[2] - self.brain_size[2])
        b_z_1 = z_begin - (numpy.random.randint(0, dif_z) if dif_z != 0 else 0)
        b_z_1 = min(max(0, b_z_1), shape[2] - self.brain_size[2])

        roi_brain = [b_x_0, b_y_0, b_z_0, self.brain_size[0], self.brain_size[1], self.brain_size[2]]
        roi_local = [x_begin - roi_brain[0], y_begin - roi_brain[1], z_begin - roi_brain[2], r_w_x, r_w_y, r_w_z]

        roi_brain_m = [shape[0] - b_x_1 - self.brain_size[0], b_y_1, b_z_1, self.brain_size[0], self.brain_size[1], self.brain_size[2]]
        roi_local_m = [shape[0] - x_begin - r_w_x - roi_brain_m[0], y_begin - roi_brain_m[1], z_begin - roi_brain_m[2], r_w_x, r_w_y, r_w_z]


        # d1 = sample["data"][0][
        #      roi_brain[0]: roi_brain[0] + roi_brain[3],
        #      roi_brain[1]: roi_brain[1] + roi_brain[4],
        #      roi_brain[2]: roi_brain[2] + roi_brain[5]]
        # d2 = sample["data"][1][
        #      roi_brain_m[0]: roi_brain_m[0] + roi_brain_m[3],
        #      roi_brain_m[1]: roi_brain_m[1] + roi_brain_m[4],
        #      roi_brain_m[2]: roi_brain_m[2] + roi_brain_m[5]]
        # d1s = d1[roi_local[0]: roi_local[0] + roi_local[3], roi_local[1]: roi_local[1] + roi_local[4], roi_local[2]: roi_local[2] + roi_local[5]]
        # d2s = d2[roi_local_m[0]: roi_local_m[0] + roi_local_m[3], roi_local_m[1]: roi_local_m[1] + roi_local_m[4], roi_local_m[2]: roi_local_m[2] + roi_local_m[5]]
        #
        # d1c = d1s[
        #       self.kernel_size // 2: -(self.kernel_size // 2): self.kernel_stride,
        #       self.kernel_size // 2: -(self.kernel_size // 2): self.kernel_stride,
        #       self.kernel_size // 2: -(self.kernel_size // 2): self.kernel_stride
        # ]
        #
        # d2c = d2s[
        #       self.kernel_size // 2: -(self.kernel_size // 2): self.kernel_stride,
        #       self.kernel_size // 2: -(self.kernel_size // 2): self.kernel_stride,
        #       self.kernel_size // 2: -(self.kernel_size // 2): self.kernel_stride
        #       ]
        # v1 = numpy.sum(d1 - d2[::-1, :, :])
        # v2 = numpy.sum(d1s - d2s[::-1, :, :])
        # v3 = numpy.sum(d1c - d2c[::-1, :, :])

        sample["roi_brain"] = numpy.array([roi_brain, roi_brain_m])
        sample["roi_local"] = numpy.array([roi_local, roi_local_m])



        return sample
# c = SelfFlip(crop_size=[159 + 2 + 2 + 2 - 4, 154 + 1 + 2 - 4, 195 + 2 - 4], brain_size=[159 + 2 + 2 + 2 - 4, 154 + 1 + 2 - 4, 195 + 2 - 4])
# c.test(opt_stride=3, kernel_size_head_local=5)