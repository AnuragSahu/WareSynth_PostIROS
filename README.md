# Synthetic_warehouse
Synthetic warehouse development, rendering and applications

# Instructions to generate Layouts

* In order to generate the warehouse along with the front and top layouts just 24 images:
```
bash genrateTestDataset # for small dataset
bash genrateDataset # for Large dataset
```

* If you want to also Generate Kitti, goto Constants files and make the GENERATE_KITTI flag true
```
gedit scripts/pillared_warehouse/Constants.py
GENERATE_KITTI = True
```

* The number of images and other configs of the warehouse can be altered by changing variables in the bottom-most for loop.

## Setup
The scripts here need blender to run, and you need to have the files set up as:
```bash
├── Project Directory
│   ├── Blender
│   │   ├── blender (executable)
│   ├── Synthetic_warehouse (This repo)
│   │   ├── scripts
│   │   ├── objects
|   |   |   ├── primitives
```



# Update on 15th July, 2020

* Added 2 new scripts- rackmakerrotation.py and warehousepillarednew.py
* Added 2 new models - Forklift and Conveyer Belt
* Created a new folder boxesavinash - for the modified boxes

# Update on 15th August, 2020

* Modularized the code. Now changes can be made to different objects in the warehouse without affecting other objects.

# How to Generate the warehouse

* Go to ./scripts/pillared_warehouse/warehouse_pillared_new.py and run it. This should generate the warehouse.

* You can make change positions of different objects by going to its respective script in the same folder.

# Update on 27th August, 2020

* User can now enter the number of racks along the length and width of the warehouse. Based on the input, the warehouse will automatically scale itself in the required directions.


