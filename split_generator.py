import random

labelIds = [i for i in range(2295)]
random.shuffle(labelIds)
train_split = labelIds[ : 1500]
val_split = labelIds[1500 : ]

print(len(train_split), len(val_split))

train_file = open("./train_files.txt", "w")
val_file = open("./val_files.txt", "w")

for i in train_split:
    train_file.write(str(i))
    if(i != train_split[-1]):
         train_file.write("\n")

for i in val_split:
    val_file.write(str(i))
    if(i != val_split[-1]):
         val_file.write("\n")

train_file.close()
val_file.close()

