import numpy as np
from glumpy import gloo, gl
from program import Program
import opengl_utils
import pvm


class MeshDrawer(object):

    # Shader type
    FLAT = 1
    PHONG = 2

    FLAT_VERTEX = """
    uniform vec4 u_color_scale;
    uniform mat4 u_view_model_normal_matrix;
    uniform mat4 u_view_model_matrix;
    //uniform mat4 u_view;
    //uniform mat4 u_model;
    uniform mat4 u_projection;
    in vec3 a_vertex;
    in vec4 a_color;
    in vec3 a_normal;
    out vec4 v_color;
    out vec4 v_position;
    out vec4 v_cs_position;
    out vec3 v_normal;
    void main()
    {
        v_color = u_color_scale * a_color;
        //vec4 cs_position = u_view * u_model * vec4(a_vertex, 1.0);
        vec4 cs_position = u_view_model_matrix * vec4(a_vertex, 1.0);
        //mat3 view_model_normal_matrix = transpose(inverse(mat3(u_view_model_matrix)));
        //vec3 cs_normal = normalize(view_model_normal_matrix * a_normal);
        vec3 cs_normal = vec3(u_view_model_normal_matrix * vec4(a_normal, 0));
        //vec3 cs_normal = a_normal;
        v_cs_position = cs_position;
        v_normal = cs_normal;
        vec4 screen_position = u_projection * cs_position;
        v_position = screen_position;
        gl_Position = screen_position;
    }
    """

    FLAT_FRAGMENT = """
    uniform float u_depth_scale;
    uniform vec3 u_normal_scale;
    in vec4 v_color;
    in vec4 v_position;
    in vec4 v_cs_position;
    in vec3 v_normal;
    void main()
    {
        float z_distance = abs(v_cs_position.z);
        float z_depth = u_depth_scale * z_distance;
        vec3 normal = u_normal_scale * v_normal;
        gl_FragData[0] = v_color;
        gl_FragData[1] = vec4(z_depth, z_depth, z_depth, 1);
        gl_FragData[2] = vec4(normal, 1);
    }
    """

    PHONG_VERTEX = """
    // Color uniforms
    uniform vec4 u_color_scale;

    // Model view projection uniforms
    uniform mat4 u_view_model_normal_matrix;
    uniform mat4 u_view_model_matrix;
    //uniform mat4 u_view;
    //uniform mat4 u_model;
    uniform mat4 u_projection;

    // Light uniforms
    uniform vec3 u_light_position;

    // Vertex attributes
    layout(location = 0) in vec3 a_vertex;
    layout(location = 1) in vec4 a_color;
    layout(location = 2) in vec3 a_normal;

    // Outputs to fragment shader
    out vec4 v_color;
    out vec4 v_screen_position;
    out vec4 v_cs_position;
    out vec3 v_cs_normal;
    out vec4 v_cs_light_position;

    void main()
    {
        v_color = u_color_scale * a_color;
        //vec4 cs_position = u_view * u_model * vec4(a_vertex, 1.0);
        v_cs_position = u_view_model_matrix * vec4(a_vertex, 1.0);
        // TODO: This should be done as a uniform
        v_cs_light_position = u_view_model_matrix * vec4(u_light_position, 1.0);
        //mat3 view_model_normal_matrix = transpose(inverse(mat3(u_view_model_matrix)));
        v_cs_normal = vec3(u_view_model_normal_matrix * vec4(a_normal, 0));
        v_screen_position = u_projection * v_cs_position;
        gl_Position = v_screen_position;
    }
    """

    PHONG_FRAGMENT = """
    #define USE_HALF_VECTOR 0
    #define USE_DERIVATIVE_FOR_NORMAL 1

    uniform float u_depth_scale;
    uniform vec3 u_normal_scale;

    // Light uniforms
    //uniform vec3 u_light_intensity;
    uniform vec3 u_light_ambient_intensity;
    uniform vec3 u_light_diffuse_intensity;
    uniform vec3 u_light_specular_intensity;

    // Material uniforms
    uniform vec3 u_material_ambient;
    uniform vec3 u_material_diffuse;
    uniform vec3 u_material_specular;
    uniform float u_material_shininess;

    in vec4 v_color;
    in vec4 v_position;
    in vec4 v_cs_position;
    in vec3 v_cs_normal;
    in vec4 v_cs_light_position;

    layout(location = 0) out vec4 out_color;
    layout(location = 1) out vec4 out_depth;
    layout(location = 2) out vec4 out_normal;

    vec3 ambient_light()
    {
        return u_material_ambient * u_light_ambient_intensity;
    }

    vec3 diffuse_light(in vec3 normal, in vec3 light_direction)
    {
        normal = normalize(normal);
        light_direction = normalize(light_direction);
        // calculation as for Lambertian reflection
        float diffuse_term = clamp(dot(normal, light_direction), 0, 1) ;
        return u_material_diffuse * u_light_diffuse_intensity * diffuse_term;
    }

    vec3 specular_light(in vec3 normal, in vec3 light_direction, in vec3 position)
    {
        normal = normalize(normal);
        light_direction = normalize(light_direction);

        float specular_term = 0.0;
#if USE_HALF_VECTOR
        vec3 v = normalize(-position);
        vec3 h = normalize(v + light_direction);
        if (dot(normal, light_direction) > 0) {
            specular_term = pow(max(dot(h, normal), 0.0), u_material_shininess); 
        }
#else
        vec3 v = normalize(-position);
        vec3 r = reflect(-light_direction, normal);
        if (dot(normal, light_direction) > 0) {
            specular_term = pow(max(dot(r, v), 0.0), u_material_shininess);
        }
#endif

        return u_material_specular * u_light_specular_intensity * specular_term;
    }

    void main()
    {
#if USE_DERIVATIVE_FOR_NORMAL
        vec3 dFdxPos = dFdx(v_cs_position.xyz);
        vec3 dFdyPos = dFdy(v_cs_position.xyz);
        vec3 cs_normal = normalize(cross(dFdxPos, dFdyPos));
#else
        vec3 cs_normal = v_cs_normal;
#endif

        // Vector to light source
        vec3 cs_light_distance_vec = v_cs_light_position.xyz - v_cs_position.xyz;
        // Calculate the cosine of the angle of incidence (brightness)
        float brightness = dot(cs_normal, cs_light_distance_vec) /
                            (length(cs_light_distance_vec) * length(cs_normal));
        brightness = max(min(brightness, 1.0), 0.0);

        vec3 cs_light_direction = v_cs_light_position.xyz - v_cs_position.xyz;
        vec3 ambient_light_vec = ambient_light();
        vec3 diffuse_light_vec = diffuse_light(cs_normal, cs_light_direction);
        vec3 specular_light_vec = specular_light(cs_normal, cs_light_direction, v_cs_position.xyz);
        out_color.xyz = v_color.xyz * (ambient_light_vec + diffuse_light_vec + specular_light_vec);

        // TODO: Vectors should be dehomogenized (i.e. w = 1)
        float z_distance = abs(v_cs_position.z);

        //out_color = v_color * brightness * vec4(u_light_intensity, 1);
        out_color.a = v_color.a;
        out_depth = vec4(u_depth_scale * vec3(z_distance, z_distance, z_distance), 1);
        out_normal = vec4(u_normal_scale * cs_normal, 1);
    }
    """

    def __init__(self, mesh, use_depth_test=True, use_face_culling=True, shader_type=PHONG):
        self.use_depth_test = use_depth_test
        self.use_face_culling = use_face_culling
        # Compile shaders
        if shader_type == self.FLAT:
            self._program = Program(self.FLAT_VERTEX, self.FLAT_FRAGMENT, version="330")
        elif shader_type == self.PHONG:
            self._program = Program(self.PHONG_VERTEX, self.PHONG_FRAGMENT, version="330")
        else:
            raise RuntimeError("Unknown shader type: {}".format(shader_type))
        self._shader_type = shader_type
        # Get mesh attributes
        self._vertices_vbo = np.core.records.fromarrays([mesh.vertices], dtype=[('a_vertex', np.float32, (3,))]).view(gloo.VertexBuffer)
        if mesh.colors is None:
            print("Mesh has no color. Using default color [0.5, 0, 0, 1].")
            mesh_colors = np.zeros((self._vertices_vbo.shape[0], 4), dtype=np.float32)
            mesh_colors[:, :] = [0.5, 0, 0, 1]
        else:
            mesh_colors = mesh.colors
        self._colors_vbo = np.core.records.fromarrays([mesh_colors], dtype=[('a_color', np.float32, (4,))]).view(gloo.VertexBuffer)
        if mesh.normals is None:
            print("Mesh has no normals. Using default normals [0, 0, 0].")
            mesh_normals = np.zeros((self._vertices_vbo.shape[0], 3), dtype=np.float32)
            mesh_normals[:, :] = [0, 0.5, 0]
        else:
            mesh_normals = mesh.normals
        self._normals_vbo = np.core.records.fromarrays([mesh_normals], dtype=[('a_normal', np.float32, (3,))]).view(gloo.VertexBuffer)
        # self._texcoords_vbo = np.core.records.fromarrays([mesh.texcoords], dtype=[('a_texcoords', np.float32, (2,))]).view(gloo.VertexBuffer)
        self._faces = mesh.faces.view(gloo.IndexBuffer)

        # Bind vertex attributes
        self._program.bind(self._vertices_vbo)
        self._program.bind(self._colors_vbo)
        self._program.bind(self._normals_vbo)
        # self._program.bind(self._texcoords_vbo)
        # Create and fill index buffer
        self._gl_index_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._gl_index_buffer)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self._faces.nbytes, self._faces, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        # Initialize uniforms
        self._color_scale = np.array([1, 1, 1, 1])
        self._normal_scale = np.array([1, 1, 1])
        self._depth_scale = 1.0
        self._light_position = np.array([-25, -25, 50.])
        self._light_intensity = 3
        self._material = np.array([0.5, 0.5, 0.5])
        self._transform = pvm.ModelTransform()

    @property
    def light_position(self):
        return self._light_position

    @light_position.setter
    def light_position(self, light_position):
        assert light_position.ndim == 1
        assert len(light_position) == 3
        self._light_position = light_position

    @property
    def light_intensity(self):
        return self._light_intensity

    @light_intensity.setter
    def light_intensity(self, light_intensity):
        self._light_intensity = light_intensity

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, material):
        assert material.ndim == 1
        assert len(material) == 3
        self._material = material

    @property
    def color_scale(self):
        return self._color_scale

    @color_scale.setter
    def color_scale(self, color_scale):
        if not isinstance(color_scale, np.ndarray) and not isinstance(color_scale, list):
            self._color_scale[:] = color_scale
        else:
            assert color_scale.ndim == 1
            assert len(color_scale) == 4
            self._color_scale = color_scale

    @property
    def normal_scale(self):
        return self._normal_scale

    @normal_scale.setter
    def normal_scale(self, normal_scale):
        if not isinstance(normal_scale, np.ndarray) and not isinstance(normal_scale, list):
            self._normal_scale[:] = normal_scale
        else:
            assert normal_scale.ndim == 1
            assert len(normal_scale) == 3
            self._normal_scale = normal_scale

    @property
    def depth_scale(self):
        return self._depth_scale

    @depth_scale.setter
    def depth_scale(self, depth_scale):
        self._depth_scale = depth_scale

    @property
    def transform(self):
        return self._transform

    @property
    def model_matrix(self):
        return self._model

    def update_transform(self, transform):
        self._transform = transform

    def draw(self, projection, view, width, height):
        model_matrix = self._transform.matrix

        # Set uniforms
        self._program["u_projection"] = projection
        # Note: OpenGL matrix multiplication works on column-major oriented storage (as least for pre-multiplication).
        # Also glumpy.glm is using column-major assumption for its operations.
        view_model_matrix = np.transpose(np.matmul(np.transpose(view), np.transpose(model_matrix)))
        self._program["u_view_model_matrix"] = view_model_matrix
        # self._program["u_view"] = view
        # self._program["u_model"] = self._model
        view_model_normal_matrix = np.transpose(np.linalg.inv(view_model_matrix))
        self._program["u_view_model_normal_matrix"] = view_model_normal_matrix
        self._program["u_color_scale"] = self._color_scale
        self._program["u_normal_scale"] = self._normal_scale
        self._program["u_depth_scale"] = self._depth_scale
        if self._shader_type == self.PHONG:
            self._program["u_light_position"] = self._light_position
            self._program["u_light_ambient_intensity"] = 0.4 * self._light_intensity
            self._program["u_light_diffuse_intensity"] = 0.4 * self._light_intensity
            self._program["u_light_specular_intensity"] = 0.2 * self._light_intensity
            self._program["u_material_ambient"] = self._material
            self._program["u_material_diffuse"] = self._material
            self._program["u_material_specular"] = self._material
            self._program["u_material_shininess"] = 32

        with self._program.activate():
            # Bind index buffer and draw
            if self.use_depth_test:
                gl.glEnable(gl.GL_DEPTH_TEST)
            else:
                gl.glDisable(gl.GL_DEPTH_TEST)
            if self.use_face_culling:
                gl.glFrontFace(gl.GL_CCW)
                gl.glCullFace(gl.GL_BACK)
                gl.glEnable(gl.GL_CULL_FACE)
            else:
                gl.glDisable(gl.GL_CULL_FACE)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._gl_index_buffer)
            gl.glDrawElements(gl.GL_TRIANGLES, 3 * len(self._faces), opengl_utils.get_gl_type(self._faces), None)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
            # gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3 * len(self._faces))
            # self._program.draw(gl.GL_TRIANGLES, self._faces)

    def tick(self, dt):
        pass
