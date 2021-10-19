import numpy
import cv2
import SimpleITK as sitk
import nibabel as nib
nib.Nifti1Header.quaternion_threshold = - numpy.finfo(numpy.float32).eps * 10


class MedicalImagePreprocess:

    def ImageRegistration(fixed, moving, other_moving = None):
        def command_iteration(method):
            pass
            # print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
            #                                        method.GetMetricValue(),
            #                                        method.GetOptimizerPosition()))

        fixed = sitk.GetImageFromArray(fixed)
        moving = sitk.GetImageFromArray(moving)

        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsCorrelation()
        # R.SetMetricAsMeanSquares()
        R.SetOptimizerAsRegularStepGradientDescent(learningRate=0.3, minStep=1e-4,
                                                   numberOfIterations=100,
                                                   gradientMagnitudeTolerance=1e-8)
        R.SetOptimizerScalesFromIndexShift()
        tx = sitk.CenteredTransformInitializer(fixed, moving, sitk.Similarity3DTransform())
        R.SetInitialTransform(tx)
        R.SetInterpolator(sitk.sitkLinear)
        R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
        outTx = R.Execute(fixed, moving)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(outTx)


        out = sitk.GetArrayFromImage(resampler.Execute(moving))
        
        if other_moving is not None:
            other_out = sitk.GetArrayFromImage(resampler.Execute(sitk.GetImageFromArray(other_moving)))
        else:
            other_out = None

        return out, other_out 
    
    def Morphology(image, kernelSize, filterFun):


        filter = filterFun()
        filter.SetKernelRadius(kernelSize)
        filter.SetForegroundValue(1)
        image = sitk.GetImageFromArray(image)
        imageed = filter.Execute(image)
        imageed = sitk.GetArrayFromImage(imageed)
        return imageed
        
        
    def ThresholdCrop(imageNp, exp = 5):
        imageSITK = sitk.GetImageFromArray(imageNp)
        inside_value = 0
        outside_value = 255
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(sitk.OtsuThreshold(imageSITK, inside_value, outside_value))
        bounding_box = label_shape_filter.GetBoundingBox(outside_value)
        bounding_box = list(bounding_box)
        bounding_box[0] = max(0, bounding_box[0] - exp)
        bounding_box[1] = max(0, bounding_box[1] - exp)
        bounding_box[2] = max(0, bounding_box[2] - exp)

        bounding_box[3] = min(imageNp.shape[2], bounding_box[0] + bounding_box[3] + exp * 2) - bounding_box[0]
        bounding_box[4] = min(imageNp.shape[1], bounding_box[1] + bounding_box[4] + exp * 2) - bounding_box[1]
        bounding_box[5] = min(imageNp.shape[0], bounding_box[2] + bounding_box[5] + exp * 2) - bounding_box[2]
        # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
        imageSITKCrop = sitk.RegionOfInterest(imageSITK, bounding_box[int(len(bounding_box) / 2):], bounding_box[0:int(len(bounding_box) / 2)])
        bounding_box = list(bounding_box)
        # bounding_box[0], bounding_box[1], bounding_box[2] = bounding_box[2], bounding_box[0], bounding_box[1]
        # bounding_box[3], bounding_box[4], bounding_box[5] = bounding_box[5], bounding_box[3], bounding_box[4]

        bounding_box[0], bounding_box[2] = bounding_box[2], bounding_box[0]
        bounding_box[3], bounding_box[5] = bounding_box[5], bounding_box[3]
        return sitk.GetArrayFromImage(imageSITKCrop), list(bounding_box)

    def Pad2Square(imageNp):
        shape = imageData.shape
        maxValue = max(shape)
        imageDataEx = numpy.zeros([maxValue, maxValue, maxValue], imageData.dtype)
        dx = (maxValue - shape[0]) // 2
        dy = (maxValue - shape[1]) // 2
        dz = (maxValue - shape[2]) // 2
        imageDataEx[dx: dx + shape[0], dy: dy + shape[1], dz: dz + shape[2]] = imageData
        return imageDataEx

    def RotateImage(imageNp):
        pass

    def AdaptiveHistogramEqualizationImageFilter(self):
        sitk.AdaptiveHistogramEqualization()

    def LabelColor(imageLabel):
        labelITK = sitk.GetImageFromArray(imageLabel)
        labelITK = sitk.LabelToRGB(labelITK)
        return sitk.GetArrayFromImage(labelITK)

    def TranslationTransform(image, offset, default_value = 0.0):
        imageSITK = sitk.GetImageFromArray(image)
        translation = sitk.TranslationTransform(len(offset), offset)


        reference_image = imageSITK
        interpolator = sitk.sitkCosineWindowedSinc

        imageSITKed = sitk.Resample(imageSITK, reference_image, translation, interpolator, default_value)

        return sitk.GetArrayFromImage(imageSITKed)

    def AffineTransform(image, scale, axis, radian, offset, center, interpolato = sitk.sitkNearestNeighbor):
        # scale = 1 退出是怎么回事？
        if radian == 0:
            return image

        imageSITKX = sitk.GetImageFromArray(image)
        affineTransform = sitk.Similarity3DTransform(scale, axis, radian, offset, center)
        imageSITKed = sitk.Resample(imageSITKX, imageSITKX, affineTransform, interpolato, 0)
        return sitk.GetArrayFromImage(imageSITKed)



    def GaussianSmooth(imageNp, sigma):
        if sigma == 0:
            return imageNp

        smooth = sitk.SmoothingRecursiveGaussianImageFilter()
        smooth.SetSigma(sigma)
        imageSITK = sitk.GetImageFromArray(imageNp)
        imageSITKed = smooth.Execute(imageSITK)
        return sitk.GetArrayFromImage(imageSITKed)

    #pu=pd+(pd−pc)(k1r2+k2r4+k3r6+…)
    def BoundedTransformation(image, k1 = 0.00000001, k2 = 0.0000000000001, distortion_center = None, interpolato = sitk.sitkNearestNeighbor):

        if k1 == 0 and k2 == 0 and numpy.count_nonzero(distortion_center) == 0:
            return image

        imageSITK = sitk.GetImageFromArray(image)
        c = distortion_center
        if not c:
            c = numpy.array(imageSITK.TransformContinuousIndexToPhysicalPoint(numpy.array(imageSITK.GetSize()) / 2.0))

        # Compute the vector image (p_d - p_c)
        delta_image = sitk.PhysicalPointSource(sitk.sitkVectorFloat64, imageSITK.GetSize(),
                                               imageSITK.GetOrigin(), imageSITK.GetSpacing(), imageSITK.GetDirection())

        delta_image_list = [sitk.VectorIndexSelectionCast(delta_image, i) - c[i] for i in range(len(c))]

        # Compute the radial distortion expression
        r2_image = sitk.NaryAdd([img ** 2 for img in delta_image_list])
        r3_image = r2_image ** (3 / 2)
        r5_image = r2_image ** (5 / 2)
        disp_image = k1 * r3_image + k2 * r5_image
        displacement_image = sitk.Compose([disp_image * img for img in delta_image_list])

        displacement_field_transform = sitk.DisplacementFieldTransform(displacement_image)
        imageSITKed = sitk.Resample(imageSITK, imageSITK, displacement_field_transform, interpolato, 0.0, imageSITK.GetPixelID())
        return sitk.GetArrayFromImage(imageSITKed)

    def Normalize3D(imageNp):
        imageNpNoZero = imageNp[imageNp > 0]
        mean = numpy.mean(imageNpNoZero)
        dev = numpy.sqrt(numpy.mean(imageNpNoZero * imageNpNoZero) - mean * mean)
        return (imageNp - mean) / dev * (imageNp > 0)
    
    def Resize3D(image, scale):
        imageSITK = sitk.GetImageFromArray(image)
        affine = sitk.AffineTransform(3)
        affine.Scale(scale)
        image_resample = sitk.Resample(imageSITK, imageSITK, affine, sitk.sitkCosineWindowedSinc, 0)
        image = sitk.GetImageFromArray(image_resample)
        return image

if __name__ == '__main__':
    pathImage = "/Users/zyx/Data/MICCAI-2012-Multi-Atlas-Challenge-Data/testing-images/1023_3.nii.gz"
    pathImage = "/Users/zyx/Data/MICCAI-2012-Multi-Atlas-Challenge-Data/training-images/1002_3.nii.gz"
    pathLabel = "/Users/zyx/Data/MICCAI-2012-Multi-Atlas-Challenge-Data/testing-labels/1024_3_glm.nii.gz"

    print(help(sitk.SmoothingRecursiveGaussianImageFilter))

    image = nib.load(pathImage)
    imageData = image.get_data()
    imageData = imageData.astype(numpy.float32) / numpy.max(imageData)
    imageData = MedicalImagePreprocess.ThresholdCrop(imageData)
    imageData = MedicalImagePreprocess.Pad2Square(imageData)

    s = 300
    imageData = numpy.zeros([s, s], numpy.uint8)
    for x in range(0, 300, 6):
        for y in range(0, 300, 6):
            cv2.circle(imageData, (x, y), 1, 255, -1)


    imageDataed = MedicalImagePreprocess.BoundedTransformation(imageData, -0.00000001, -0.0000000000001, -0.0000000000001)

    #！
    for i, image in enumerate(imageData):
        cv2.imshow("a", image)
        cv2.imshow("10", imageDataed)
        # cv2.imshow("20", imageData20[i])
        # cv2.imshow("30", imageData30[i])
        # cv2.imshow("40", imageData40[i])

        cv2.waitKey(0)