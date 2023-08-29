import os
import numpy as np
import nibabel as nib

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

class ImageProcessor:
    def __init__(self, img_type, data_dir):
        self.img_type = img_type
        self.data_dir = data_dir

    def process_images(self):
        data_files = glob.glob(self.data_dir)

        for i, data in enumerate(data_files):
            npz_file = np.load(data)
            keys = npz_file.files
            dataset = npz_file['image']
            labels = npz_file['label']
            image_data = dataset[:]
            label_data = labels[:]
            
            print("===============================")
            print("Image number", i)
            print(data)
            print("Image sum:", image_data.sum())
            print("Image max:", image_data.max())
            print("Label max:", labels.max())

            # Save image and label visualizations
            plt.imsave("img.png", image_data, cmap='gray')
            plt.imsave("label.png", label_data, cmap='bone')

            image1 = cv2.imread('img.png')
            image2 = cv2.imread('label.png')

            height1, width1, _ = image1.shape
            height2, width2, _ = image2.shape

            combined_height = max(height1, height2)
            combined_width = width1 + width2

            combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

            combined_image[:height1, :width1] = image1
            combined_image[:height2, width1:] = image2

            cv2.imwrite('viz/combined_image_{}.png'.format(i), combined_image)

            print("Processed and saved visualization.")

# Example usage



class NiftiProcessor:
    def __init__(self, nii_img_path, nii_label_path):
        self.nii_img_path = nii_img_path
        self.nii_label_path = nii_label_path
        self.nifti_img = nib.load(nii_img_path)
        self.nifti_label = nib.load(nii_label_path)
        self.nifti_data = self.nifti_img.get_fdata()
        self.labels = self.nifti_label.get_fdata()

    def extract_patient_name(self):
        last_slash_index = self.nii_img_path.rfind('/')
        first_dot_index = self.nii_img_path.find('.')
        patient = self.nii_img_path[last_slash_index + 1:first_dot_index]
        return patient

    def process_and_save_slices(self, output_dir, axis_to_split=2):
        os.makedirs(output_dir, exist_ok=True)

        slices_list = []
        labels_list = []

        for slice_index in range(self.nifti_data.shape[axis_to_split]):
            if axis_to_split == 0:
                slice_data = self.nifti_data[slice_index, :, :]
            elif axis_to_split == 1:
                slice_data = self.nifti_data[:, slice_index, :]
            else:
                slice_data = self.nifti_data[:, :, slice_index]
                label_data = self.labels[:, :, slice_index]

            slices_list.append(slice_data)
            labels_list.append(label_data)

        slices_array = np.array(slices_list)
        labels_array = np.array(labels_list)

        patient = self.extract_patient_name()

        for i, (image, label) in enumerate(zip(slices_array, labels_array)):
            min_value = np.min(image)
            max_value = np.max(image)
            image = (image - min_value) / (max_value - min_value)

            np.savez(os.path.join(output_dir, "{}_slice_{}".format(patient, i)), image=image, label=labels_array[i])

        print("Slices and labels saved in npz files.")


# Example usage
nii_img_path = '/data/BRATS_2018/HGG/Brats18_2013_2_1/Brats18_2013_2_1_t2.nii.gz'
nii_label_path = '/data/BRATS_2018/HGG/Brats18_2013_2_1/Brats18_2013_2_1_seg.nii.gz'
output_dir = "npz"
img_type = "npz"
data_dir = "npz/*"
slicer = NiftiProcessor(nii_img_path, nii_label_path)
slicer.process_and_save_slices(output_dir)


showimages = ImageProcessor(img_type, data_dir)
showimages.process_images()
