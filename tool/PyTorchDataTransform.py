import numpy
import tool.MedicalImagePreprocess as mp
import batchgenerators.transforms.spatial_transforms as st


class SpatialTransform:
    def __init__(self, patch_size, patch_center_dist_from_border=30,
                 do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                 do_rotation=True, angle_x=(0, 2 * numpy.pi), angle_y=(0, 2 * numpy.pi), angle_z=(0, 2 * numpy.pi),
                 do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                 border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, data_key="data",
                 label_key="seg", p_el_per_sample=1, p_scale_per_sample=1, p_rot_per_sample=1):
        
        self.transform = st.SpatialTransform(patch_size = patch_size,
                                          patch_center_dist_from_border=patch_center_dist_from_border,
                                          do_elastic_deform=do_elastic_deform,
                                          alpha=alpha,
                                          sigma=sigma,
                                          do_rotation=do_rotation,
                                          angle_x=angle_x,
                                          angle_y=angle_y,
                                          angle_z=angle_z,
                                          do_scale=do_scale,
                                          scale=scale,
                                          border_mode_data=border_mode_data,
                                          border_cval_data=border_cval_data,
                                          order_data=order_data,
                                          border_mode_seg=border_mode_seg,
                                          border_cval_seg=border_cval_seg,
                                          order_seg=order_seg,
                                          random_crop=random_crop,
                                          data_key=data_key,
                                          label_key=label_key,
                                          p_el_per_sample=p_el_per_sample,
                                          p_scale_per_sample=p_scale_per_sample,
                                          p_rot_per_sample=p_rot_per_sample)
    
    def __call__(self, sample):
        sample = self.transform(**sample)
        
class RandomFlip:
    def __call__(self, sample):
        image = sample["image"]

        if numpy.random.uniform(-1., 1.) > 0:
            image = image[::, ::, ::-1]

        if numpy.random.uniform(-1., 1.) > 0:
            image = image[::-1, ::, ::]

        sample["image"] = image

        return sample

class FixedCrop:

    def __init__(self, fixed_crop_size):
        self.fixed_crop_size = fixed_crop_size

    def __call__(self, sample):
        image = sample["image"]

        begin0 = (image.shape[0] - self.fixed_crop_size[0]) // 2
        begin1 = (image.shape[1] - self.fixed_crop_size[1]) // 2
        begin2 = (image.shape[2] - self.fixed_crop_size[2]) // 2


        sample["image"] = image[
                begin0: begin0 + self.fixed_crop_size[0],
                begin1: begin1 + self.fixed_crop_size[1],
                begin2: begin2 + self.fixed_crop_size[2]]

        return sample

class RandomCrop:

    def __init__(self, fixed_crop_size, rate = 0.5):
        self.fixed_crop_size = fixed_crop_size
        self.rate = rate

    def __call__(self, sample):
        image = sample["image"]
        w0, w1, w2 = image.shape

        begin0 = (image.shape[0] - self.fixed_crop_size[0]) // 2
        begin1 = (image.shape[1] - self.fixed_crop_size[1]) // 2
        begin2 = (image.shape[2] - self.fixed_crop_size[2]) // 2

        x0 = begin0 + numpy.random.randint(-begin0 * self.rate, begin0 * self.rate)
        x1 = begin1 + numpy.random.randint(-begin1 * self.rate, begin1 * self.rate)
        x2 = begin2 + numpy.random.randint(-begin2 * self.rate, begin2 * self.rate)

        sample["image"] = image[x0: x0 + self.fixed_crop_size[0], x1: x1 + self.fixed_crop_size[1], x2: x2 + self.fixed_crop_size[2]]

        return sample



class NormalizeToTensor:

    def __call__(self, sample):

        attribute = sample["image"]


        sample["image"] = numpy.expand_dims(sample["image"], axis=0).copy()
        sample["label"] = numpy.expand_dims(sample["label"], axis=0)

        return sample