from bpy import context
import bpy
from mathutils import Matrix,Vector
from mathutils.geometry import normal
# from Assets import assets
# from CameraProperties import cameraProperties
import numpy as np
from bpy_extras.object_utils import world_to_camera_view


class FOV(object):

    def camera_as_planes(self, scene, obj):
        """
        Return planes in world-space which represent the camera view bounds.
        """

        camera = obj.data
        # normalize to ignore camera scale
        matrix = obj.matrix_world.normalized()
        frame = [matrix @ v for v in camera.view_frame(scene=scene)]
        origin = matrix.to_translation()

        planes = []
        is_persp = (camera.type != 'ORTHO')
        for i in range(4):
            # find the 3rd point to define the planes direction
            if is_persp:
                frame_other = origin
            else:
                frame_other = frame[i] + matrix.col[2].xyz

            n = normal(frame_other, frame[i - 1], frame[i])
            d = -n.dot(frame_other)
            planes.append((n, d))

        if not is_persp:
            # add a 5th plane to ignore objects behind the view
            n = normal(frame[0], frame[1], frame[2])
            d = -n.dot(origin)
            planes.append((n, d))

        return planes


    def side_of_plane(self, p, v):
        return p[0].dot(v) + p[1]


    def is_segment_in_planes(self, p1, p2, planes):
        dp = p2 - p1

        p1_fac = 0.0
        p2_fac = 1.0

        for p in planes:
            div = dp.dot(p[0])
            if div != 0.0:
                t = -self.side_of_plane(p, p1)
                if div > 0.0:
                    # clip p1 lower bounds
                    if t >= div:
                        return False
                    if t > 0.0:
                        fac = (t / div)
                        p1_fac = max(fac, p1_fac)
                        if p1_fac > p2_fac:
                            return False
                elif div < 0.0:
                    # clip p2 upper bounds
                    if t > 0.0:
                        return False
                    if t > div:
                        fac = (t / div)
                        p2_fac = min(fac, p2_fac)
                        if p1_fac > p2_fac:
                            return False

        ## If we want the points
        # p1_clip = p1.lerp(p2, p1_fac)
        # p2_clip = p1.lerp(p2, p2_fac)        
        return True


    def point_in_object(self, obj, pt):
        xs = [v[0] for v in obj.bound_box]
        ys = [v[1] for v in obj.bound_box]
        zs = [v[2] for v in obj.bound_box]
        pt = obj.matrix_world.inverted() @ pt
        return (min(xs) <= pt.x <= max(xs) and
                min(ys) <= pt.y <= max(ys) and
                min(zs) <= pt.z <= max(zs))


    def object_in_planes(self, obj, planes):
        
        matrix = obj.matrix_world
        box = [matrix @ Vector(v) for v in obj.bound_box]
        for v in box:
            if all(self.side_of_plane(p, v) > 0.0 for p in planes):
                # one point was in all planes
                return True

        # possible one of our edges intersects
        edges = ((0, 1), (0, 3), (0, 4), (1, 2),
                (1, 5), (2, 3), (2, 6), (3, 7),
                (4, 5), (4, 7), (5, 6), (6, 7))
        if all(self.is_segment_in_planes(box[e[0]], box[e[1]], planes)
            for e in edges):
            return False

        return False


    def objects_in_planes(self, objects, planes, origin):
        """
        Return all objects which are inside (even partially) all planes.
        """
        return [obj for obj in objects
                if self.point_in_object(obj, origin) or
                self.object_in_planes(obj, planes)]

    def select_objects_in_camera(self):
        scene = context.scene
        origin = scene.camera.matrix_world.to_translation()
        planes = self.camera_as_planes(scene, scene.camera)
        objects_in_view = self.objects_in_planes(scene.objects, planes, origin)

        objects_in_fov = []

        for obj in objects_in_view:
            objects_in_fov.append(obj.name)
        
        return objects_in_fov

    def project_3d_point(self, camera: bpy.types.Object,
                         p: Vector,
                         render: bpy.types.RenderSettings = bpy.context.scene.render) -> Vector:
        """
        Given a camera and its projection matrix M;
        given p, a 3d point to project:

        Compute P’ = M * P
        P’= (x’, y’, z’, w')

        Ignore z'
        Normalize in:
        x’’ = x’ / w’
        y’’ = y’ / w’

        x’’ is the screen coordinate in normalised range -1 (left) +1 (right)
        y’’ is the screen coordinate in  normalised range -1 (bottom) +1 (top)

        :param camera: The camera for which we want the projection
        :param p: The 3D point to project
        :param render: The render settings associated to the scene.
        :return: The 2D projected point in normalized range [-1, 1] (left to right, bottom to top)
        """

        if camera.type != 'CAMERA':
            raise Exception("Object {} is not a camera.".format(camera.name))

        if len(p) != 3:
            raise Exception("Vector {} is not three-dimensional".format(p))

        # Get the two components to calculate M
        modelview_matrix = camera.matrix_world.inverted()
        projection_matrix = camera.calc_matrix_camera(
            bpy.data.scenes["Scene"].view_layers["View Layer"].depsgraph,
            x = render.resolution_x,
            y = render.resolution_y,
            scale_x = render.pixel_aspect_x,
            scale_y = render.pixel_aspect_y,
        )

        # print(projection_matrix * modelview_matrix)

        # Compute P’ = M * P
        p1 = projection_matrix @ modelview_matrix @ Vector((p.x, p.y, p.z, 1))

        # Normalize in: x’’ = x’ / w’, y’’ = y’ / w’
        p2 = Vector(((p1.x/p1.w, p1.y/p1.w)))

        # project_points in camera frame
        proj_p_pixels = Vector(((render.resolution_x -1)*(p2.x+1)/2,
                                (render.resolution_y -1)*(p2.y-1)/(-2)))
        return p2, proj_p_pixels

    def object_in_frame(self, projected_points,
                        render: bpy.types.RenderSettings = bpy.context.scene.render):
        
        max_x = render.resolution_x
        max_y = render.resolution_y
        
#        print(max_x, max_y)
#        print(projected_points[0], projected_points[1][1], end=" Here\n")
        if(projected_points[1][0] < max_x and projected_points[1][1] < max_y and
           projected_points[1][0] > 0 and projected_points[1][1] > 0):
            return True
        else:
            return False

    def get_objects_in_fov(self):
        possible_obj_in_fov = self.select_objects_in_camera()
#        print("Objects in FOV : ", possible_obj_in_fov)
        objects_in_FOV = []
        shelfs_count = 0

        # Old Way
        # for shelf in possible_obj_in_fov:
        #     lineup = "LineUp"
        #     linedown = "LineDown"
        #     if shelf == "Shelf":
        #         pass
        #     else:
        #         lineup +=shelf[-4:]
        #         linedown +=shelf[-4:]
        #     shelfs_count += 1
        #     if lineup in possible_obj_in_fov and linedown in possible_obj_in_fov:
        #         objects_in_FOV.append(shelf)
        #     elif linedown not in possible_obj_in_fov and lineup not in possible_obj_in_fov:
        #         continue
        #     else:
        #         return ["invalid"]
        # if shelfs_count < 2:
        #     return ["invalid"]
        # return objects_in_FOV

        # New Way
        for shelf in possible_obj_in_fov:
            obj = bpy.data.objects[shelf]
            points = self.project_3d_point(camera = context.scene.camera,
                                            p = obj.location,
                                            render = bpy.context.scene.render)
            if(self.object_in_frame(points) or (shelf[:5] == "Shelf") and points[1][1] > 0):
                objects_in_FOV.append(shelf)
        
        return objects_in_FOV


fov = FOV()

if __name__ == "__main__":
    objects = fov.get_objects_in_fov()
    print(objects)