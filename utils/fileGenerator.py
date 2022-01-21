import glob

#########################################
# TO BE CONFIGURED FROM DATA GENERATION TEAM
# NUMBER OF IMAGES IN ONE SEQUENCE
number_of_images_per_sequence = 51
TRAIN_SEQ = 180
#########################################

# put the path of scratch folder
mypath = "/home/anurag/Research/Unity/combine_data/finalData/img/*"
train_file_name = "train_temporal_files.txt"
test_file_name = "test_temporal_files.txt"

# Get the list of all sequences in 
seqLists = glob.glob(mypath)
seqLists.sort()

# How many sequences are present?
lenSeqLists = len(seqLists)

number_of_sequences = int(lenSeqLists/number_of_images_per_sequence)

seqNum = 0

# make train file
train_file = open(train_file_name, 'w+')
for i in range(TRAIN_SEQ):
    for j in range(number_of_images_per_sequence):
        train_file.write(seqLists[i*number_of_images_per_sequence + j] + "\n")
    train_file.write(",")

train_file.close()


# make train file
test_file = open(test_file_name, 'w+')
for i in range(TRAIN_SEQ, number_of_sequences):
    for j in range(number_of_images_per_sequence):
        test_file.write(seqLists[i*number_of_images_per_sequence + j] + "\n")
    test_file.write(",")

test_file.close()
 