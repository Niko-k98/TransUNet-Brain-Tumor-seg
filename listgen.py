import os

def get_files_in_directory(root_directory):
    file_list = []

    for root, dirs, files in os.walk(root_directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_path=file_path.split('.')
            file_list.append(file_path[0])

    return file_list

# Example usage
root_directory = "/data/Koutsoubn8/Bratz_2018/HGG/train_npz"
output_file = 'lists/lists_Bratz/HGG/train.txt'
file_list = get_files_in_directory(root_directory)
with open(output_file, "w") as f:
    for file_path in file_list:
        f.write(file_path + "\n")
        print(file_path)
root_directory = "/data/Koutsoubn8/Bratz_2018/HGG/test_vol"
output_file = 'lists/lists_Bratz/HGG/test_vol.txt'
file_list = get_files_in_directory(root_directory)
with open(output_file, "w") as f:
    for file_path in file_list:
        f.write(file_path + "\n")
        print(file_path)
