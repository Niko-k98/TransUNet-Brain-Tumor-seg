
import h5py
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2

img_type= "npz"
# file='data/Synapse/train_npz/case0005_slice020.npz'

data_dir=glob.glob("../data/bratz/train_npz/Brats18_TCIA01_401_1_t1_slice_14.npz")

# print(data_dir)
# Open the HDF5 file
for i,data in enumerate(data_dir):
    npz_file = np.load(data)
    keys=npz_file.files
    dataset = npz_file['image']
    labels=npz_file['label']
    image_data = dataset[:]
    label_data=labels[:]
    print(image_data.sum())
    # exit()
    print("===============================")
    print("image number", i)
    print(data)
    # print(label_data.sum())
    # print(image_data.shape)
    print("img",image_data.max())
    print("lbl",labels.max())

    # Display the image using matplotlib
    plt.imsave("img.png" ,image_data, cmap='gray')
    # cv2.imwrite("img.png",image_data)
          # Assuming a grayscale image
    # plt.imsave("label_viz/label_{}.png".format(i) ,label_data[1], cmap='gray')  # Assuming a grayscale image
    plt.imsave("label.png",label_data,cmap='bone')
    # cv2.imwrite('label.png',label_data)
    # plt.axis('off')  # Turn off axis labels
    # plt.show()
    image1 = cv2.imread('img.png')
    image2 = cv2.imread('label.png')
                
                # Make sure all images have the same dimensions
    height1, width1, _ = image1.shape
    height2, width2, _ = image2.shape

    # Determine the size of the combined image
    combined_height = max(height1, height2)
    combined_width = width1 + width2

    # Create an empty canvas for the combined image
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    # Paste the first image on the left and the second image on the right
    combined_image[:height1, :width1] = image1
    combined_image[:height2, width1:] = image2

    # Save or display the combined image
    # cv2.imwrite("combined_image.jpg", combined_image)

    cv2.imwrite('viz/combined_image_{}.png'.format(i), combined_image)


# Close the HDF5 file
# h5_file.close()
