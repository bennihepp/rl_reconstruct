import sys
import numpy as np
import glumpy
# import cv2
from glumpy import gloo, glm, gl
from plyfile import PlyData, PlyElement

width = 640
height = 480
height = 640

vertex = """
uniform vec4 u_color;
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
attribute vec3 a_position;
attribute vec4 a_color;
varying vec4 v_color;
varying vec4 v_position;
void main()
{
    v_color = u_color * a_color;
    vec4 out_position = u_projection * u_view * u_model * vec4(a_position,1.0);
    gl_Position = out_position;
    //gl_Position = <transform>;
    v_position = gl_Position;
}
"""

fragment = """
//in vec4 ShadowCoord;
varying vec4 v_color;
varying vec4 v_position;
void main()
{
    float n = length(v_position);
    float gray_level = 1 - n/10;
    //gl_FragColor = vec4(vec3(gray_level), 1);
    //gl_FragColor = v_color;
    gl_FragData[0] = v_color;
    gl_FragData[1] = vec4(vec3(gray_level), 1);
}
"""

from glumpy.geometry import colorcube
cube = gloo.Program(vertex, fragment)
vertices, faces, outline = colorcube()
vertices = vertices.view([('a_position', np.float32, (3,)),
                          ('a_texcoord', np.float32, (2,)),
                          ('a_normal', np.float32, (3,)),
                          ('a_color', np.float32, (4,))])
cube.bind(vertices)
cube['u_model'] = np.eye(4, dtype=np.float32)
cube['u_view'] = glm.translation(0, 0, -5)
phi, theta = 0, 0

vertex = """
#version 330

uniform float time;

attribute vec2 position;

void main()
{
    vec2 xy = vec2(sin(2.0 * time));
    gl_Position = vec4(
        position * (0.25 + 0.75 * xy * xy),
        0.0, 1.0);
    //gl_Position = vec4(position, 0.0, 1.0);
}
"""

fragment = """
#version 330

uniform vec4 color;

//layout(location = 0) out vec4 gl_FragColor;

void main() {
    gl_FragColor = color;
}
"""

quad = gloo.Program(vertex, fragment)
quad2 = gloo.Program(vertex, fragment)

quad2["position"] = np.array([(-0.5, -0.5),
                    (-0.5, +0.5),
                    (+0.5, -0.5),
                    (+0.5, +0.5)])
# quad["position"] = np.array([(-1, -1),
#                     (-1, +1),
#                     (+1, -1),
#                     (+1, +1)])
quad["position"] = np.array([(-0, -0),
                    (-0, +1),
                    (+1, -0),
                    (+1, +1)])
quad["color"] = np.array([1, 0, 0, 1]) # red
quad["time"] = np.array(0.0)

cube['u_projection'] = glm.perspective(45.0, width / float(height), 2.0, 100.0)

window_config = glumpy.app.configuration.get_default()
window_config.double_buffer = True
window_config.samples = 0
window_config.api = "ES"
window_config.major_version = 3
window_config.minor_version = 1
window_config.profile = "core"
window = glumpy.app.Window(width=width, height=height, config=window_config, visible=True)
print(window.config)

#fb = np.zeros((640, 480), np.float32).view(gloo.TextureFloat2D)
color = np.zeros((height, width, 4), np.ubyte).view(gloo.Texture2D)
normal_depth = np.zeros((height, width, 4), np.ubyte).view(gloo.Texture2D)
depth_buffer = gloo.DepthBuffer(width, height)
# color[..., 3] = 255
color.interpolation = gl.GL_LINEAR
normal_depth.interpolation = gl.GL_LINEAR
#color = np.ones((640, 480,4),np.ubyte).view(gloo.Texture2D)
framebuffer = gloo.FrameBuffer(color=[color, normal_depth], depth=depth_buffer)
#framebuffer = gloo.RenderBuffer(width=640, height=480)

q = False

data = {"time": 0}

@window.event
def on_init():
    gl.glEnable(gl.GL_DEPTH_TEST)

@window.event
def on_resize(width, height):
    cube['u_projection'] = glm.perspective(45.0, width / float(height), 2.0, 100.0)
    model = np.eye(4, dtype=np.float32)
    # transform["projection"] = model

@window.event
def on_draw(dt):
    data["time"] += dt
    quad["time"] = data["time"]
    # window.clear()
    gl.glViewport(0, 0, window.width, window.height)

    # framebuffer.activate()
    # gl.glViewport(0, 0, width, height)
    # quad["time"] = 0.2
    # quad.draw(gl.GL_TRIANGLE_STRIP)
    # quad2["time"] = 0.1
    # quad2["color"] = 0,0,1,1
    # quad2.draw(gl.GL_TRIANGLE_STRIP)
    # gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0, gl.GL_FRONT)
    # values = gl.glReadPixels(0, 0, width, height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
    # values = np.fromstring(values, np.ubyte).reshape((height, width, 4))
    # red = gl.glReadPixels(0, 0, width, height, gl.GL_RED, gl.GL_UNSIGNED_BYTE)
    # #red = np.fromstring(red, np.ubyte).reshape((480, 640))
    # #red = np.transpose(red)
    # red = np.fromstring(red, np.ubyte).reshape((height, width))
    # #red = np.transpose(red)
    # print(values.shape)
    # print(np.sum(values[..., 0]))
    # print(np.sum(values[..., 3]))
    # print(np.sum(red))
    # print(np.min(red))
    # print(np.max(red))
    # print(red[320,240])
    # img = values[..., [2, 1, 0, 3]]
    # #img = red
    # cv2.imwrite("img.png", img)
    # cv2.imshow("img", img)
    # cv2.waitKey(50)
    # framebuffer.deactivate()

    # framebuffer.color[0][:] = 0
    # framebuffer.color[1][:] = 0

    framebuffer.activate()
    gl.glViewport(0, 0, width, height)
    # gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0 | gl.GL_COLOR_ATTACHMENT1)
    gl.glDrawBuffers(np.array([gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1], dtype=np.uint32))
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

    global phi, theta
    # Filled cube
    gl.glDisable(gl.GL_BLEND)
    # gl.glDisable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
    cube['u_color'] = np.array([1, 1, 1, 1])
    cube.draw(gl.GL_TRIANGLES, faces)

    # Outlined cube
    # gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
    # gl.glEnable(gl.GL_BLEND)
    # gl.glDepthMask(gl.GL_FALSE)
    # cube['u_color'] = np.array([0, 0, 0, 1])
    # cube.draw(gl.GL_LINES, outline)
    # gl.glDepthMask(gl.GL_TRUE)

    gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
    # gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT1)
    pixels = gl.glReadPixels(0, 0, width, height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
    gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
    # pixels = gl.glReadPixels(0, 0, width, height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)

    framebuffer.deactivate()

    gl.glDisable(gl.GL_BLEND)
    gl.glDisable(gl.GL_DEPTH_TEST)

    # pixels = 100 * np.ones((height, width, 4), np.ubyte)
    # pixels[:, :, 3] = 1
    # pixels[:, :, 2] = 0

    gl.glDrawPixels(width, height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, pixels)

    # gl.glDisable(gl.GL_DEPTH_TEST)
    # # quad["time"] = 0.2
    # quad.draw(gl.GL_TRIANGLE_STRIP)
    # quad2["time"] = 0.1
    # quad2["color"] = 0, 0, 1, 1
    # quad2.draw(gl.GL_TRIANGLE_STRIP)
    # gl.glEnable(gl.GL_DEPTH_TEST)

    # Make cube rotate
    theta += 0.5 # degrees
    phi += 0.5 # degrees
    model = np.eye(4, dtype=np.float32)
    glm.rotate(model, theta, 0, 0, 1)
    glm.rotate(model, phi, 0, 1, 0)
    cube['u_model'] = model
    # transform["view"] = glm.translation(0, 0, -5)
    # transform["model"] = model

# args = glumpy.app.parser.get_options()
# glumpy.app.use("glfw")
#glumpy.app.use("pyglet")
# glumpy..use("sdl2")
#glumpy.app.use("pyside")
# glumpy.app.use("osxglut")

from glumpy.transforms import Trackball, Position
from glumpy.transforms import PVMProjection, Position
# transform = PVMProjection(Position("a_position"))
# transform = PVMProjection(Trackball(Position("a_position")))
# transform = Trackball(Position("a_position"))
# cube['transform'] = transform
# cube['transform'] = PVMProjection(Trackball(Position("a_position")))
# window.attach(cube['transform'])

glumpy.app.run()

import sys
sys.exit(0)

glumpy.app.__init__(backend=glumpy.app.__backend__)

# print(np.sum(fb))
# print(fb.pending_data)
framebuffer.activate()
quad.draw(gl.GL_TRIANGLE_STRIP)
framebuffer.deactivate()
framebuffer.activate()
gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0, gl.GL_FRONT)
values = gl.glReadPixels(0, 0, 640, 480, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
values = np.fromstring(values, np.ubyte).reshape((640, 480, 4))
print(values.shape)
# print(fb.pending_data)
# print(np.sum(fb))
print(np.sum(values[..., 0]))
print(np.sum(values[..., 1]))
print(np.sum(values[..., 2]))
print(np.sum(values[..., 3]))

#plydata = PlyData.read(sys.argv[1])
#for i, elements in enumerate(plydata.elements):
#    print(i, elements.name)
#    #print("  ", elements.data.keys())
#    print("  ", len(elements.data))

#print(sys.argv[1])
#mesh = obj.Obj(sys.argv[1])

# cv2.imwrite("img.png", values)

