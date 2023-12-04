import os
import numpy as np
import nibabel as nib
import random
import shutil
import h5py
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
        # print(nii_label_path)
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
            image = np.divide((image - min_value), (max_value - min_value), out=np.zeros_like(image), where=image != 0)# (image - min_value) / (max_value - min_value)
            # if np.isnan(image).any():
            #     nan_mask = np.isnan(image)
            #     image=np.where(nan_mask,0,image)

            np.savez(os.path.join(output_dir, "{}_slice_{}".format(patient, i)), image=image, label=labels_array[i])

        # print("Slices and labels saved in npz files.")




def move_folders(source, destination, percentage, move_back=False):
    if move_back:
        source, destination = destination, source
   
    folders = [folder for folder in os.listdir(source) if os.path.isdir(os.path.join(source, folder))]
    # print(len(folders))
    num_folders_to_move = int(len(folders) * (percentage / 100))
    folders_to_move = random.sample(folders, num_folders_to_move)

    for folder in folders_to_move:
        source_path = os.path.join(source, folder)
        destination_path = os.path.join(destination, folder)
        shutil.move(source_path, destination_path)
        # print(source_path)
        # print(destination_path)

    print(f"{num_folders_to_move} folders moved from {source} to {destination}.")

# def save_subdirectory_data(subdirectories, output_directory):
#     for subdir in subdirectories:
#         npz_files = glob.glob(os.path.join(subdir, "*.npz"))
#         for npz_file in npz_files:
#             data = np.load(npz_file)
#             output_npy_path = os.path.join(output_directory, f"{os.path.basename(subdir)}.npy")
#             # print(data['image'])
#             print(output_npy_path)
#             np.save(output_npy_path, (data['image'],data['label']))  # Adjust this based on your dataset
def save_subdirectory_data(subdirectories, output_directory):
    for subdir in subdirectories:
        lslash=subdir.rfind("/")
        fdot=subdir.find('.')
        h5name=subdir[lslash + 1:fdot]
        npz_files = glob.glob(os.path.join(subdir, "*.npz"))
        images=[]
        labels=[]
        print("generating h5 file for", h5name )
        for i, npz_file in enumerate(npz_files):
            npz_data = np.load(npz_file)
            image=npz_data['image']
            label=npz_data['label']
            images.append(image)
            labels.append(label)  
        image_array=np.array(images)
        label_array=np.array(labels)    
        # print(image_array.shape)
        # print(label_array.shape)
        # exit()   
        with h5py.File(output_directory +"/{}.h5".format(h5name), 'w') as h5:              
                h5.create_dataset(('image'), data=image_array)
                h5.create_dataset(('label'),data=label_array)
            
            # print("Data saved to", output_npy_path)

#process bratz data into Transunet format (npz file with (image, label) arrays) 
nii_img_path    =     "/data/BRATS_2018/HGG/*/*" # path to unprocessed img foler
nii_label_path  =     "/data/BRATS_2018/HGG/*/*seg*" #path to unprocessed labels
output_dir      =     "/data/Koutsoubn8/Bratz_2018/HGG/train_npz" # output folder for train imgs
output_test_dir =     "/data/Koutsoubn8/Bratz_2018/HGG/test_slices" # output folder for test slices
output_directory =    "/data/Koutsoubn8/Bratz_2018/HGG/test_vol"    # output folder to test volume
percentage_to_move=    20  # % of data to move to testing

#use to display image and gtruth (wip)
# data_dir = "../data/bratz/train_npz/*"
# img_type = "npz"


nii_img_path=glob.glob(nii_img_path)
filterseg=['seg.']
nii_img_path=[element for element in nii_img_path if not any(char in element for char in filterseg)]
nii_label_path=glob.glob(nii_label_path)
# print(nii_label_path)




print(len(nii_img_path))
print("="*30)
print(len(nii_label_path))

# exit()



# output_dir="/data/Koutsoubn8/Bratz_2018/HGG/train_npz" # train slices

# output_test_dir="/data/Koutsoubn8/Bratz_2018/HGG/test_slices" # test slices

root_directory = output_test_dir
# output_directory = "/data/Koutsoubn8/Bratz_2018/HGG/test_vol"
        # if not os.path.exists(output_directory):
        #     os.makedirs(output_directory, exist_ok=True)

# output_dir = "../data/bratz/train_npz"

j=0
for i, case in enumerate(nii_img_path):
    if (i) % 4==0:
        j=j+1
        patient=nii_label_path[j-1].split('/')
       
        # exit()
        print('+'*60)
    print("="*30)
    # patient=slicer.extract_patient_name
    print("i", i,'j', j)
    print("case : ",case)
    print("label: ",nii_label_path[j-1])
    try:
        slicer = NiftiProcessor(case, nii_label_path[j-1])
        print(output_dir + "/"+patient[3] + "_" + patient[4] + "/")
        slicer.process_and_save_slices(output_dir + "/"+patient[3] + "_" + patient[4] + "/")
    except:
        print("THERE WAS AN ERROR")
    # move_folders(output_dir, output_test_dir, percentage_to_move)


    # if (i+1) % 40 == 0: 
move_folders(output_dir, output_test_dir, percentage_to_move)

        # npz_directory ="..data/Koutsoubn8/Bratz_2018/HGG/test_slices"

                        # "../data/Koutsoubn8/Bratz_2018/HGG/test_slices"
        # root_directory = "../data/Koutsoubn8/Bratz_2018/HGG/test_slices"
        # output_directory = "../data/Koutsoubn8/Bratz_2018/HGG/test_vol"
    
        # output_directory = "..data/bratz/test_vol"

subdirectories = [os.path.join(root_directory, subdir) for subdir in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, subdir))]
save_subdirectory_data(subdirectories, output_directory)
            # use to stop early /visualize
        #     showimages = ImageProcessor(img_type, data_dir)
        #     showimages.process_images()
        # exit()
# move_folders(output_dir, output_test_dir, percentage_to_move)


# exit()
# slicer = NiftiProcessor(nii_img_path, nii_label_path)
# slicer.process_and_save_slices(output_dir)

#use for visualization
# img_type = "npz"
# data_dir = "../data/bratz/train_npz"
# showimages = ImageProcessor(img_type, data_dir)
# showimages.process_images()
