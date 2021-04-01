import bpy
from random import randrange
import random
import sys, os
from math import *
from mathutils import Matrix,Vector
import numpy as np
import random
from mathutils.bvhtree import BVHTree

prob_of_box = 0.5

Name = 'Dimi'


bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)


def append_zero(num):
    return "." + str(num).zfill(3)


all_box_loc = './objects/primitives/boxes_avinash/'
rack_loc = './objects/primitives/Racks/modal.dae'

offset_x = -0.5
z_positions = [0.3, 1.63, 2.98, 4.35]
y_positions = [-2.3, -1.2, 0, 1.5]

# box_count = {"BoxA": 0, "BoxB": 0, "BoxC": 0, "BoxD": 0,
#              "BoxF": 0, "BoxG": 0, "BoxH": 0, "BoxI": 0}

# distance_between_racks_x = 1
# distance_between_racks_y = 2.7


# num_rack_y = 14  # 14
# y_rack = 5.25
# y_start = -(y_rack * (num_rack_y/2.0 - 1) + y_rack/2.0)
# y_end = -y_start

# corridor = 5
# num_rack_layer_x = 14  # should be 6, 14, 22...
# x_rack = 1.5
# x_start = -(num_rack_layer_x/4.0 * x_rack +
#             (num_rack_layer_x - 2) * corridor/8.0)
# x_end = -x_start

# z = 0

# x_coord = []
# y_coord = []

# x = x_start
# while x <= x_end:
#     x_coord.append(x)
#     x += (2 * x_rack + corridor)

# y = y_start
# while y <= y_end:
#     y_coord.append(y)
#     y += y_rack

# print(x_coord)
# print(y_coord)

subscript = 0
#x_coord = x_start


def make_racks(rack_location, subscript):
    imported_object = bpy.ops.wm.collada_import(filepath=rack_loc)

    model = "Rack"
    change = append_zero(subscript)

    name = model
    # if subscript > 0:
    #     name = model + change
    # else:
    #     name = model

    bpy.data.objects[name].location.x += rack_location[0]
    bpy.data.objects[name].location.y += rack_location[1]
    bpy.data.objects[name].location.z += rack_location[2]
    boxes = ["BoxB","BoxA", "BoxC","BoxH"]
    prev_model=None
    for rows in z_positions:
        for cols in y_positions:
            if  random.randint(0,1) <= prob_of_box:
                model = random.choice(boxes)
                model_temp = model
    #            model = model+(append_zero(box_count[model_temp]))

                change = append_zero(box_count[model_temp])
                if box_count[model_temp] > 0:
                    model = model + change
                else:
                    model = model

                box_count[model_temp] += 1
                final_model_location = all_box_loc + model_temp + "/model1.dae"
                print(final_model_location)
                imported_object = bpy.ops.wm.collada_import(
                    filepath=final_model_location)
                obj = bpy.data.objects[model]
                rot_mat = Matrix.Rotation(radians(random.randint(0, 45)), 4, 'Z')
                orig_loc, orig_rot, orig_scale = obj.matrix_world.decompose()
                print(orig_loc)
                orig_loc_mat = Matrix.Translation(orig_loc)
                print(Vector((10, 10, 10)))
                orig_rot_mat = orig_rot.to_matrix().to_4x4()
                orig_scale_mat = np.dot(np.dot(Matrix.Scale(orig_scale[0],4,(1,0,0)),Matrix.Scale(orig_scale[1],4,(0,1,0))),Matrix.Scale(orig_scale[2],4,(0,0,1)))
                obj.matrix_world = np.dot(orig_loc_mat,np.dot(rot_mat,np.dot(orig_rot_mat,orig_scale_mat)))
                bpy.data.objects[model].location.x = offset_x+ rack_location[0]
                bpy.data.objects[model].location.y = cols + rack_location[1]
                bpy.data.objects[model].location.z = rows + rack_location[2]
                prev_model = model
    os.mkdir('./objects/primitives/CustomRacks/rack_' + str(subscript))
    bpy.ops.wm.collada_export(
        filepath='./objects/primitives/CustomRacks/rack_' + str(subscript) + '/model.dae')


num_racks = 1

for i in range(num_racks):

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)

    box_count = {"BoxA": 0, "BoxB": 0, "BoxC": 0, "BoxD": 0,
                 "BoxF": 0, "BoxG": 0, "BoxH": 0, "BoxI": 0}

    subscript = i
    make_racks([0, 0, 0], subscript)

# update scene, if needed
dg = bpy.context.evaluated_depsgraph_get()
dg.update()
# bpy.ops.wm.collada_export(filepath='./rendered_warehouse/'+sys.argv[4]+'.dae')
# bpy.ops.export_scene.fbx(filepath='./rendered_warehouse/' + sys.argv[4]+'.fbx', path_mode='RELATIVE