import numpy as np
import matplotlib.pyplot as plt
# Load the .npz file
file_path = '../data/bratz/train_npz/Brats18_TCIA01_401_1_t1_slice_14.npz'
data = np.load(file_path)

# List the keys (array names) stored in the .npz file
array_names = data.files
print("Arrays in the file:", array_names)
img=data['label']
print(img)

min_value = np.min(img)
max_value = np.max(img)
img = (img- min_value) / (max_value - min_value)
if np.isnan(img).any():
    nan_mask = np.isnan(img)
    img=np.where(nan_mask,0,img)
# print(img.max())
# exit()

# Access and explore individual arrays
# for array_name in array_names:
#     array = data[array_name]
#     print(f"Array '{array_name}':")
#     print("Data type:", array.dtype)
#     print("Shape:", array.shape)
#     print("Data:")
#     # print(array)
#     print(array.max())
#     print(array.sum())
#     print("=" * 30)

print(img.max())
plt.imshow(img, cmap='bone')
plt.axis('off')  # Turn off axis labels and ticks
plt.title('Grayscale Image')

# Save the image to a file
output_file = 'grayscale_image.png'
plt.savefig(output_file, bbox_inches='tight', pad_inches=0)


# Don't forget to close the file after you're done
data.close()
# min_value = np.min(nifti_data)
# max_value = np.max(nifti_data)
# normalized_data = (nifti_data - min_value) / (max_value - min_value)
