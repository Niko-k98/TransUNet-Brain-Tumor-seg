import SimpleITK as sitk
import numpy as np
import h5py

# A path to a T1-weighted brain .nii image:
# t1_fn = 'path_to_file.nii'

# # Read the .nii image containing the volume with SimpleITK:
# sitk_t1 = sitk.ReadImage(t1_fn)

# # and access the numpy array:
# t1 = sitk.GetArrayFromImage(sitk_t1)

#==================================
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
#==================================
nii_img_path= '/data/BRATS_2018/HGG/Brats18_2013_2_1/Brats18_2013_2_1_flair.nii.gz'
nii_label_path='/data/BRATS_2018/HGG/Brats18_2013_2_1/Brats18_2013_2_1_seg.nii.gz'
nifti_img  = nib.load(nii_img_path)
nifti_label = nib.load(nii_label_path)

nifti_data = nifti_img.get_fdata()
labels = nifti_label.get_fdata()

last_slash_index = nii_img_path.rfind('/')
first_dot_index = nii_img_path.find('.')

# Extract the desired portion of the path
patient = nii_img_path[last_slash_index + 1:first_dot_index]



print(nifti_data.shape)
print(nifti_label.shape)

data_dict = {
    'image': nifti_img,
    'label': nifti_label
}
file_path = 'h5_test.h5'

# Save data_dict to the HDF5 file
with h5py.File(file_path, 'w') as hdf_file:
    hdf_file.create_dataset('image', data=data_dict['image'])
    hdf_file.create_dataset('label', data=data_dict['label'])

print("Data saved to", file_path)

with h5py.File(file_path,'r') as file:
    print(file['image'].shape)
    print(file['label'].shape)



exit()

# Create a directory to store the slices
output_dir = "output_slices"
os.makedirs(output_dir, exist_ok=True)

# Create lists to store slices and labels
slices_list = []
labels_list = []

axis_to_split = 2  # Change this to the axis you want to split along

for slice_index in range(nifti_data.shape[axis_to_split]):
    if axis_to_split == 0:
        slice_data = nifti_data[slice_index, :, :]
    elif axis_to_split == 1:
        slice_data = nifti_data[:, slice_index, :]
    else:
        slice_data = nifti_data[:, :, slice_index]
        label_data = labels[:, : ,slice_index]
        # print(label_data.shape)
    

    # Append the slice and associated label to the lists
    slices_list.append(slice_data)
    # labels_list.append(labels[slice_index])  # Assuming labels are associated with slices
    labels_list.append(label_data)

# Convert lists to NumPy arrays
slices_array = np.array(slices_list)
print(slices_array.shape)
labels_array = np.array(labels_list)
# labels_array=np.transpose(labels_array,(1,2,0))
print(labels_array.shape)

# print(slices_array[0])
# Save slices and labels in an npz file

for i, (image, label)  in enumerate(zip(slices_array,labels_array)):
    min_value = np.min(image)
    max_value = np.max(image)
    image = (image- min_value) / ((max_value - min_value))
    # print(image.max())
    # print(label.max())
    # exit()
    
    np.savez("npz/{}_slice_{}".format(patient,i), image=image, label=labels_array[i])
    exit()
print("Slices and labels saved in npz file.")