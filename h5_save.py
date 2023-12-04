
import h5py
#
file_path ="data/Synapse/test_vol_h5/case0001.npy.h5"

with h5py.File(file_path, 'r') as hdf_file:
    # print(hdf_file.size)
    dataset_keys = list(hdf_file.keys())
    # Access datasets and convert to NumPy arrays
    dataset1 = (hdf_file['image'][:],hdf_file['label'][:])
    
    # You can access more datasets similarly if needed
print("Keys (Dataset Names) within the HDF5 file:")
for key in dataset_keys:
    print(key)
# Now you can use the loaded NumPy arrays (dataset1, dataset2) as needed
print("Data from dataset1:")
print(dataset1['image'])

print("Data from dataset2:")
# print(dataset2.shape)