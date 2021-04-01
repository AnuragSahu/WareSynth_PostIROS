import random

def classify(labelID):
    simple = 0
    complex = 0
    for i in labelID:
        if(i < 2000):
            simple += 1
        else:
            complex += 1
    return simple, complex

labelIds = [i for i in range(2000)] + [i for i in range(6000, 8000)]
random.shuffle(labelIds)
train_split = labelIds[ : int(0.75 * len(labelIds))]
val_split = labelIds[int(0.75 * len(labelIds)) : ]

print(len(train_split), len(val_split))

#random.shuffle(train_split)
#random.shuffle(val_split)

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

print(classify(train_split))
print(classify(val_split))
