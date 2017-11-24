import numpy as np
from glumpy import gl


NUMPY_TO_OPENGL_TYPE_DICT = {
    np.dtype(np.uint8): gl.GL_UNSIGNED_BYTE,
    np.dtype(np.uint16): gl.GL_UNSIGNED_SHORT,
    np.dtype(np.uint32): gl.GL_UNSIGNED_INT,
    np.dtype(np.float32): gl.GL_FLOAT}


def get_gl_type(obj):
    if isinstance(obj, np.ndarray):
        gl_type = NUMPY_TO_OPENGL_TYPE_DICT[obj.dtype]
    elif isinstance(obj, np.dtype):
        gl_type = NUMPY_TO_OPENGL_TYPE_DICT[obj]
    else:
        raise ValueError("Invalid object to determine OpenGL type")
    return gl_type
