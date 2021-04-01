import bpy, json
import numpy as np
from mathutils import Matrix,Vector

def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    #T_world2bcam = -1*R_world2bcam * location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
        ))
    return np.array(RT)

cam = bpy.data.objects['Camera']
#active = bpy.context.active_object
#selected = bpy.context.selected_objects

all_data = []
for obj in bpy.data.objects:
#    obj.dimensions = active.dimensions
    
    entity = {}
    
    print("Name:")
    print(obj.name)
    
    if obj.name == "Camera":
        continue
    elif "Box" in obj.name:
        label = "box"
    elif "Shelf" in obj.name:
        label = "shelf"
    
    print("\nLocation:")
    print(obj.location)
    
    print("\nDimension:")
    print(obj.dimensions)
    
    print("\nScale:")
    print(obj.scale)

    RT = get_3x4_RT_matrix_from_blender(cam)
    extra_vec = np.array([0, 0, 0, 1])
    
    RT = np.vstack((RT, extra_vec))
    print(RT)
    
    
    all_coords = np.array([])
    
    print("\nBounding Box:")
    for idx in range(8):
#        print(obj.bound_box[idx][0])
#        print(obj.scale[0])
        
        x = obj.bound_box[idx][0] * obj.scale[0]
        y = obj.bound_box[idx][1] * obj.scale[1]
        z = obj.bound_box[idx][2] * obj.scale[2]
        
#        print(x, y, z)
        coord = np.array([x, y, z, 1], dtype='f')
        
#        print(coord)
#        print(RT @ coord)
        
        transformed_coords = RT @ coord
        transformed_coords /= transformed_coords[3]
        
        transformed_coords = transformed_coords[:3]
        
        print(transformed_coords)
        
#        print(all_coords.shape[0] == 0)
        if all_coords.shape[0] != 0:
            all_coords = np.vstack((all_coords, transformed_coords))
        else:
            all_coords = transformed_coords
            
        print("")
        
    entity['label'] = label
    entity['x'] = obj.location[0]
    entity['y'] = obj.location[1]
    entity['z'] = obj.location[2]
    
    entity['sz_x'] = obj.dimensions[0]
    entity['sz_y'] = obj.dimensions[1]
    entity['sz_z'] = obj.dimensions[2]
    
    entity['bbox'] = all_coords.tolist()
    
    all_data.append(entity)
    
with open('train.json', 'w') as f:
    f.write(json.dumps(all_data, indent=4, sort_keys=True))