import os
import shutil

# Get all the files in the parent directory
parent_directory = "/home/anurag/Research/Unity/combine_data_10"

d = parent_directory
dirs = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]

dirs.sort()

# Create a new folder 
new_path = parent_directory + "/afinalData"
os.mkdir(new_path)
os.mkdir(new_path + "/Annotations")
os.mkdir(new_path + "/Correspondences")
os.mkdir(new_path + "/depth")
os.mkdir(new_path + "/img")
os.mkdir(new_path + "/Keypoints")
os.mkdir(new_path + "/debugOutputs")
os.mkdir(new_path + "/keypointImages")
os.mkdir(new_path + "/topLayouts")

for dir in dirs:
    batch_number = int(dir.split("_")[-1])
    # get the path to all the folders
    Annotations = dir + "/Annotations"
    Correspondences = dir + "/Correspondences"
    depth = dir + "/depth"
    img = dir + "/img"
    Keypoints = dir + "/Keypoints"

    new_Annotations = new_path + "/Annotations"
    new_Correspondences = new_path + "/Correspondences"
    new_depth = new_path + "/depth"
    new_img = new_path + "/img"
    new_Keypoints = new_path + "/Keypoints"
    
    # get the name of all the files in annotations
    all_files = [ f.path for f in os.scandir(Annotations) ]
    
    # extract the name of file
    for file in all_files:
        file_name = file.split("/")[-1].split(".")[0]
        new_file_name = str(batch_number)+"_"+file_name
        
        source = Annotations + "/" + file_name + ".txt"
        destination = new_Annotations + "/" + new_file_name + ".txt"
        shutil.move(source, destination)

        source = Correspondences + "/" + file_name + ".txt"
        destination = new_Correspondences + "/" + new_file_name + ".txt"
        shutil.move(source, destination)

        source = depth + "/" + file_name + ".png"
        destination = new_depth + "/" + new_file_name + ".png"
        shutil.move(source, destination)

        source = img + "/" + file_name + ".png"
        destination = new_img + "/" + new_file_name + ".png"
        shutil.move(source, destination)

        source = Keypoints + "/" + file_name + ".txt"
        destination = new_Keypoints + "/" + new_file_name + ".txt"
        shutil.move(source, destination)
