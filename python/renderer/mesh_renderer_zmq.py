import os
import numpy as np
import glumpy
from glumpy import gloo, gl, glm
from pybh import log_utils
from pybh import zmq_utils
from pybh import utils
from pybh import camera_utils
from pybh import math_utils
from pybh.utils import argparse_bool
from framebuffer import Framebuffer
from trackball import Trackball
from mesh_drawer import MeshDrawer
from mesh import SimpleMesh, CubeMesh
from application import Application
from framebuffer_drawer import FramebufferDrawer
from window import Window
import renderer_zmq_service


logger = log_utils.get_logger("RLrecon/mesh_renderer")


# import pydevd
# pydevd.settrace('localhost', port=1234, stdoutToServer=True, stderrToServer=True)


def run(args):
    # Command line arguments
    address = args.address
    show_window = args.show_window
    poll_timeout = args.poll_timeout
    width = args.width
    height = args.height
    mesh_filename = args.mesh_filename
    use_msgpack_for_mesh = args.use_msgpack_for_mesh
    window_width = args.window_width
    if window_width is None:
        window_width = width
    window_height = args.window_height
    if window_height is None:
        window_height = height
    window_pos_x = args.window_pos_x
    window_pos_y = args.window_pos_y

    model_scale = args.model_scale
    model_rpy = [args.model_roll, args.model_pitch, args.model_yaw]
    depth_scale = args.depth_scale
    yaw_speed = args.yaw_speed
    pitch_speed = args.pitch_speed

    # Compute vertical field of view (needed for glm.perspective)
    horz_fov = args.horz_fov
    focal_length = camera_utils.fov_to_focal_length(math_utils.degrees_to_radians(horz_fov), width)
    vert_fov = math_utils.radians_to_degrees(camera_utils.focal_length_to_fov(focal_length, height))
    logger.info("Horizontal FOV: {:.4f}, vertical FOV: {:.4f}".format(horz_fov, vert_fov))
    window_focal_length = camera_utils.fov_to_focal_length(math_utils.degrees_to_radians(horz_fov), window_width)
    window_vert_fov = math_utils.radians_to_degrees(camera_utils.focal_length_to_fov(window_focal_length, window_height))
    logger.info("Window vertical FOV: {:.4f}".format(window_vert_fov))

    # Fixed parameters
    initial_distance = 10  # initial camera distance

    # Glumpy initialization
    # glumpy.app.use("glfw")
    # glumpy.app.use("glfw", 'ES', 3, 3)
    glumpy_config = glumpy.app.configuration.get_default()
    glumpy_config.double_buffer = True
    glumpy_config.samples = 0
    glumpy_config.api = "ES"
    glumpy_config.depth_size = 32
    glumpy_config.major_version = 3
    glumpy_config.minor_version = 1
    glumpy_config.profile = "core"

    # if show_window:
    window = Window(width=window_width, height=window_height, fov=window_vert_fov, znear=0.5, zfar=1000, config=glumpy_config, visible=show_window,
                    title=b"mesh_renderer_zmq")
    trackball = Trackball(pitch=0, yaw=0)
    window.glumpy_window.attach(trackball)

    app = Application()
    logger.info("Glumpy backend: {}".format(glumpy.app.__backend__))

    # Process initial events
    if window_pos_x is not None or window_pos_y is not None:
        if window_pos_x is None:
            window_pos_x, _ = window.glumpy_window.get_position()
        if window_pos_y is None:
            _, window_pos_y = window.glumpy_window.get_position()
        window.glumpy_window.set_position(window_pos_x, window_pos_y)

    # Offscreen surface initialization
    framebuffer = Framebuffer(width, height, num_color_attachments=3, color_dtypes=[np.ubyte, np.float32, np.float32])
    fb_color_drawer = FramebufferDrawer(framebuffer, color_index=0, draw_mode=FramebufferDrawer.BLIT)
    fb_depth_drawer = FramebufferDrawer(framebuffer, color_index=1, draw_mode=FramebufferDrawer.DRAW_QUAD)
    fb_depth_drawer.set_color_scale(depth_scale)
    fb_normal_drawer = FramebufferDrawer(framebuffer, color_index=2, draw_mode=FramebufferDrawer.DRAW_QUAD)
    fb_normal_drawer.set_color_scale(0.5)
    fb_normal_drawer.set_color_offset(0.5)

    # Offscreen surface initialization
    window_framebuffer = Framebuffer(window_width, window_height, num_color_attachments=3, color_dtypes=[np.ubyte, np.float32, np.float32])
    window_fb_color_drawer = FramebufferDrawer(window_framebuffer, color_index=0, draw_mode=FramebufferDrawer.BLIT)
    window_fb_depth_drawer = FramebufferDrawer(window_framebuffer, color_index=1, draw_mode=FramebufferDrawer.DRAW_QUAD)
    window_fb_depth_drawer.set_color_scale(depth_scale)
    window_fb_normal_drawer = FramebufferDrawer(window_framebuffer, color_index=2, draw_mode=FramebufferDrawer.DRAW_QUAD)
    window_fb_normal_drawer.set_color_scale(0.5)
    window_fb_normal_drawer.set_color_offset(0.5)

    # Add framebuffer draw for normals
    window.add_drawer(window_fb_normal_drawer)

    # Transformation from world coordinates (z up, x foward, y left) to opengl coordinates (z backward, x right, y up)
    world_to_opengl_mat = np.eye(4, dtype=np.float32)
    world_to_opengl_mat = glm.xrotate(world_to_opengl_mat, -90)
    world_to_opengl_mat = glm.zrotate(world_to_opengl_mat, 90)
    opengl_to_world_mat = np.linalg.inv(world_to_opengl_mat)

    # Mesh loading
    if mesh_filename is not None:
        mesh = None
        if use_msgpack_for_mesh:
            msgpack_mesh_filename = mesh_filename + ".msgpack"
            if os.path.isfile(msgpack_mesh_filename):
                logger.info("Loading mesh from msgpack file {}".format(msgpack_mesh_filename))
                mesh = SimpleMesh.read_from_msgpack(msgpack_mesh_filename)
        if mesh is None:
            logger.info("Loading mesh from file {}".format(mesh_filename))
            mesh = SimpleMesh.read_from_file(mesh_filename)
            if use_msgpack_for_mesh:
                logger.info("Saving mesh to msgpack file {}".format(msgpack_mesh_filename))
                mesh.write_to_msgpack(msgpack_mesh_filename)
    else:
        mesh = CubeMesh()
    logger.info("Setting uniform color")
    mesh.set_colors_uniform([0.5, 0.5, 0.5, 1])
    # logger.info("Computing mesh colors with z-colormap")
    # mesh.set_colors_with_coordinate_colormap(min_coord=0, max_coord=10)

    # Drawer initialization
    # cube_mesh = CubeMesh()
    # cube_drawer = MeshDrawer(cube_mesh)
    # cube_drawer.depth_scale = 0.1
    # cube_drawer.normal_scale = 0.1
    # mesh = SimpleMesh.read_from_ply(sys.argv[1])
    # mesh.write_to_pickle("mesh.pickle")
    # mesh = SimpleMesh.read_from_pickle("mesh2.pickle")
    # mesh = Mesh.from_file(filename)
    # mesh_drawer = MeshDrawer(mesh)
    mesh_drawer = MeshDrawer(mesh)
    mesh_drawer.transform.scale = model_scale
    mesh_drawer.transform.orientation_rpy = model_rpy
    # Assume mesh is in world coordinate system
    # mesh_drawer.model = np.matmul(np.transpose(world_to_opengl_mat), mesh_drawer.model)
    # mesh_drawer.color_scale = color_scale
    # mesh_drawer.normal_scale = normal_scale
    # mesh_drawer.depth_scale = depth_scale
    # glm.scale(mesh_drawer.model, 0.01)
    # cube_drawer = CubeDrawer()

    # Server initialization
    logger.info("Starting ZMQ server on address {}".format(address))
    server_conn = zmq_utils.Connection(address, zmq_utils.zmq.REP)
    server_conn.bind()
    renderer_service = renderer_zmq_service.RendererZMQService(
        framebuffer, mesh_drawer, vert_fov, initial_distance, trackball, world_to_opengl_mat)

    global input_enabled
    global override_renderer_service_transform
    global window_enabled
    input_enabled = args.input_enabled
    window_enabled = window.visible
    override_renderer_service_transform = False

    def window_visible_callback(visible):
        if visible:
            logger.info("Showing window")
            window.show()
        else:
            logger.info("Hiding window")
            window.hide()

    def window_active_callback(active):
        global window_enabled
        if active:
            logger.info("Activating window")
        else:
            logger.info("Deactivating window")
        window_enabled = active

    renderer_service.window_visible_callbacks.register(window_visible_callback)
    renderer_service.window_active_callbacks.register(window_active_callback)

    # Some keyboard input handling
    @window.glumpy_window.event
    def on_key_press(symbol, modifiers):
        try:
            character = chr(symbol).lower()
        except ValueError as err:
            character = ""
        logger.debug("Key pressed: {}, modifiers: {}, character: {}".format(symbol, modifiers, character))
        if character == 'n':
            logger.info("Showing normals")
            window.clear_drawers()
            window.add_drawer(window_fb_normal_drawer)
        elif character == 'c':
            logger.info("Showing colors")
            window.clear_drawers()
            window.add_drawer(window_fb_color_drawer)
        elif character == 'd':
            logger.info("Showing depth")
            window.clear_drawers()
            window.add_drawer(window_fb_depth_drawer)
        elif character == 'i':
            global input_enabled
            if input_enabled:
                logger.info("Disabling input")
                input_enabled = False
            else:
                logger.info("Enabling input")
                input_enabled = True
        elif character == 'a':
            global window_enabled
            if window_enabled:
                logger.info("Disabling window")
                window_enabled = False
            else:
                logger.info("Enabling window")
                window_enabled = True
        elif character == 'o':
            global override_renderer_service_transform
            if override_renderer_service_transform:
                logger.info("Not overriding renderer service transform input")
                override_renderer_service_transform = False
            else:
                logger.info("Overriding renderer service transform input")
                override_renderer_service_transform = True
        elif character == 'p':
            logger.info("Current location: {}".format(trackball.location))
            logger.info("Current world location: {}".format(np.dot(opengl_to_world_mat[:3, :3], trackball.camera_location)))
            logger.info("Current orientation_rpy: {}".format(trackball.orientation_rpy))

    timer = utils.RateTimer(reset_interval=500)
    while True:
        dt = app.clock.tick()

        # Handle network requests
        request_dump = server_conn.recv(timeout=poll_timeout * 1000)
        if request_dump is not None:
            response_dump = renderer_service.handle_request_msg(request_dump, raise_exception=True)
            if response_dump is not None:
                server_conn.send(response_dump)

        # Perform any drawer processing
        mesh_drawer.tick(dt)

        # Animate model if desired
        if yaw_speed != 0:
            mesh_drawer.transform.yaw += yaw_speed * dt
        if pitch_speed != 0:
            mesh_drawer.transform.pitch += pitch_speed * dt

        if override_renderer_service_transform:
            renderer_service.view_transform = trackball.transform.copy()

        if window.visible and window_enabled:
            if input_enabled and renderer_service.input_enabled:
                view_transform = trackball.transform
            else:
                view_transform = renderer_service.view_transform
            with window_framebuffer.activate(clear=True):
                mesh_drawer.draw(window.projection, view_transform.matrix, window.width, window.height)

        # For debugging
        # color_pixels = framebuffer.read_rgba_pixels()
        # depth_pixels = framebuffer.read_rgba_pixels(color_index=1)
        # import cv2
        # cv2.imshow("color", color_pixels)
        # cv2.imshow("depth", depth_pixels[:, :, 0])
        # cv2.waitKey(50)

        window.override_view(trackball.view)
        app.process(dt)

        timer.update_and_print_rate(print_interval=100)


def main():
    _, non_glumpy_args = glumpy.app.parser.get_default().parse_known_args()

    import argparse

    parser = argparse.ArgumentParser(description='Simple mesh renderer.')
    parser.add_argument('--show-window', type=argparse_bool, default=True,
                        help='Show window with rendered content.')
    parser.add_argument('--input-enabled', type=argparse_bool, default=True)
    parser.add_argument('--address', type=str, default="tcp://*:22222",
                        help='Address to bind socket to.')
    parser.add_argument('--width', type=int, default=640,
                        help='Width of render surface.')
    parser.add_argument('--height', type=int, default=480,
                        help='Height of render surface.')
    parser.add_argument('--window-width', type=int)
    parser.add_argument('--window-height', type=int)
    parser.add_argument('--window-pos-x', type=int)
    parser.add_argument('--window-pos-y', type=int)
    parser.add_argument('--horz-fov', type=float, default=90,
                        help='Horizontal field of view.')
    parser.add_argument('--poll-timeout', type=float, default=0.01,
                        help='Poll timeout for ZMQ sockets to keep event loop alive.')
    parser.add_argument('--mesh-filename', type=str,
                        help='Mesh to render.')
    parser.add_argument('--use-msgpack-for-mesh', type=argparse_bool, default=False)
    parser.add_argument('--yaw-speed', type=float, default=0.0)
    parser.add_argument('--pitch-speed', type=float, default=0.0)

    parser.add_argument('--model-scale', type=float, default=1.0)
    parser.add_argument('--model-roll', type=float, default=0.0)
    parser.add_argument('--model-pitch', type=float, default=0.0)
    parser.add_argument('--model-yaw', type=float, default=0.0)
    parser.add_argument('--depth-scale', type=float, default=0.02)

    args = parser.parse_args(non_glumpy_args)

    run(args)


if __name__ == "__main__":
    from pybh import thread_utils
    thread_utils.enable_thread_profiling("zmq_threads.cprof")
    main()
