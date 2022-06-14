# make the vix in blender 
import bpy


class CREATE_VIZ_BLENDER:
    def __init__(self):
        self.BB_3D_Boxes = []
        self. BB_3D_Shelves = []
    def read_annotations(self, path):
        ann_file = open(path, 'r')
        # first read boxes then shelves
        for ann in ann_file:
            type, x, y, z, length, width, height = ann.split(',')
            if(type == "Box"):
                self.BB_3D_Boxes.append([x, y, z,length, width, height])
            else:
                self.BB_3D_Shelves.append([x, y, z,length, width, height])     

    
    def make_in_blender(self):
        # For all the boxes
        for box in self.BB_3D_Boxes:
            x,y,z,length, width, height = box
            x = int(x)
            y = int(y)
            z = int(z)
            length = int(length)
            width = int(width)
            height = int(height)

            x = x - 269
            y = y - 256
            z = z - 256
            
            location = [x, y, z] 
            scale = [length/2, width/2, height/2]
            # call the cube making function
            bpy.ops.mesh.primitive_cube_add(location=location,scale=scale)
        for shelf in self.BB_3D_Shelves:
            x,y,z,length, width, height = shelf
            x = int(x)
            y = int(y)
            z = int(z)
            length = int(length)
            width = int(width)
            height = int(height)
            
            x = x - 269
            y = y - 256
            z = z - 256

            location = [x, y, z]
            scale = [length/2, width/2, height/2]
            # call the cube making function
            bpy.ops.mesh.primitive_cube_add(location = location, scale = scale)

viz = CREATE_VIZ_BLENDER()
viz.read_annotations("/Users/vampire/RESEARCH/ann.txt")
viz.make_in_blender()