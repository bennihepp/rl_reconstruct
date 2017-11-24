import numpy as np
from pybh import serialization


class SimpleMesh(object):

    def __init__(self, vertices, faces, colors=None, normals=None):
        self._vertices = np.asarray(vertices, dtype=np.float32)
        assert self._vertices.ndim == 2
        assert self._vertices.shape[1] == 3
        self._faces = np.asarray(faces, dtype=np.uint32)
        assert self._faces.ndim == 2
        assert self._faces.shape[1] == 3
        assert np.min(self._faces) >= 0
        assert np.max(self._faces) < self._vertices.shape[0]
        if colors is None:
            self._colors = None
        else:
            self._colors = np.asarray(colors, dtype=np.float32)
            assert self._colors.ndim == 2
            assert self._colors.shape[1] == 4
        if normals is None:
            self._normals = None
        else:
            self._normals = np.asarray(normals, dtype=np.float32)
            assert self._normals.ndim == 2
            assert self._normals.shape[1] == 3
        # self._texcoords = None

    def set_colors_uniform(self, color):
        self._colors = np.empty((self._vertices.shape[0], 4), dtype=np.float32)
        self._colors[:] = color

    def set_colors_with_coordinate_colormap(self, coordinate_axis=1, min_coord=None, max_coord=None):
        self._colors = np.empty((self._vertices.shape[0], 4), dtype=np.float32)
        if min_coord is None:
            min_coord = np.min(self._vertices[:, coordinate_axis])
        if max_coord is None:
            max_coord = np.max(self._vertices[:, coordinate_axis])

        def color_from_normalized_coord(coord):
            r = np.minimum(2 * coord, 1)
            g = 0.5 * np.ones(coord.shape)
            b = np.maximum(2 * (coord - 0.5), 0)
            a = 1 * np.ones(coord.shape)
            return np.stack([r, g, b, a], axis=1)
        normalized_coord = (self._vertices[:, coordinate_axis] - min_coord) / (max_coord - min_coord)
        self._colors[:, :] = color_from_normalized_coord(normalized_coord)

    @staticmethod
    def read_from_file(filename):
        import pymesh
        mesh_data = pymesh.load_mesh(filename)
        vertices = mesh_data.vertices.astype(dtype=np.float)
        faces = mesh_data.faces
        rgba_attribute_names = ["vertex_red", "vertex_green", "vertex_blue", "vertex_alpha"]
        normal_attribute_names = ["vertex_nx", "vertex_ny", "vertex_nz"]
        if all([mesh_data.has_attribute(attr) for attr in rgba_attribute_names]):
            print("Mesh has color")
            color_dtype = mesh_data.get_attribute_ref(rgba_attribute_names[0]).dtype
            colors = np.empty((mesh_data.num_vertices, 4), dtype=color_dtype)
            for i, attr in enumerate(rgba_attribute_names):
                colors[:, i] = mesh_data.get_attribute_ref(attr)
            if color_dtype == np.ubyte:
                colors = colors.astype(np.float32) / 255
        else:
            colors = None
        if all([mesh_data.has_attribute(attr) for attr in normal_attribute_names]):
            print("Mesh has normals")
            normals = np.empty((mesh_data.num_vertices, 3), dtype=np.float32)
            for i, attr in enumerate(normal_attribute_names):
                normals[:, i] = mesh_data.get_attribute_ref(attr)
        else:
            normals = None
        return SimpleMesh(vertices, faces, colors, normals)

    @staticmethod
    def read_from_ply(filename):
        from plyfile import PlyData, PlyElement
        plydata = PlyData.read(filename)
        num_faces = plydata["face"].count
        faces = np.empty((num_faces, 3), dtype=np.uint32)
        for i, face in enumerate(plydata["face"]["vertex_indices"]):
            faces[i, :] = face
        num_vertices = plydata["vertex"].count
        vertices = np.empty((num_vertices, 3), dtype=np.float32)
        vertices[:, 0] = plydata["vertex"]["x"]
        vertices[:, 1] = plydata["vertex"]["y"]
        vertices[:, 2] = plydata["vertex"]["z"]
        colors = np.empty((num_vertices, 4), dtype=np.float32)
        colors[:, 0] = plydata["vertex"]["red"] / 255.
        colors[:, 1] = plydata["vertex"]["green"] / 255.
        colors[:, 2] = plydata["vertex"]["blue"] / 255.
        colors[:, 3] = plydata["vertex"]["alpha"] / 255.
        return SimpleMesh(vertices, faces, colors)

    @staticmethod
    def read_from_pickle(filename):
        serializer = serialization.PickleSerializer()
        with open(filename, "rb") as file:
            mesh = serializer.load(file)
        return mesh

    def write_to_pickle(self, filename):
        serializer = serialization.PickleSerializer()
        with open(filename, "wb") as file:
            serializer.dump(self, file)

    @staticmethod
    def read_from_msgpack(filename):
        serializer = serialization.MsgPackSerializer()
        with open(filename, "rb") as file:
            mesh_dict = serializer.load(file)
        return SimpleMesh(mesh_dict["vertices"],
                          mesh_dict["faces"],
                          mesh_dict["colors"],
                          mesh_dict["normals"])

    def write_to_msgpack(self, filename):
        serializer = serialization.MsgPackSerializer()
        mesh_dict = {
            "vertices": self._vertices,
            "faces": self._faces,
            "colors": self._colors,
            "normals": self._normals,
        }
        with open(filename, "wb") as file:
            serializer.dump(mesh_dict, file)

    @property
    def vertices(self):
        return self._vertices

    @property
    def faces(self):
        return self._faces

    def has_colors(self):
        return self._colors is not None

    @property
    def colors(self):
        return self._colors

    def has_normals(self):
        return self._colors is not None

    @property
    def normals(self):
        return self._normals


class CubeMesh(SimpleMesh):

    def __init__(self):
        from glumpy.geometry import colorcube
        vertices, faces, outline = colorcube()
        faces = faces.reshape((-1, 3))
        super(CubeMesh, self).__init__(vertices["position"], faces, vertices["color"], vertices["normal"])
