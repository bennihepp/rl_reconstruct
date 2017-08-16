import pytest
import numpy as np
import cv2
from RLrecon.engines.unreal_cv_wrapper import UnrealCVWrapper


@pytest.fixture(scope="module")
def engine():
    print("Creating UnrealCVWrapper")
    cv_wrapper = UnrealCVWrapper()
    print("Yielding UnrealCVWrapper")
    yield cv_wrapper
    print("Closing UnrealCVWrapper")
    cv_wrapper.close()
    print("Closed UnrealCVWrapper")


def test_rgb_image(engine, show_image=False):
    print('Engine type: ', type(engine))
    img = engine.get_rgb_image()
    assert type(img) == np.ndarray
    assert img.shape[0] == 480
    assert img.shape[1] == 640
    assert img.shape[2] == 3
    if show_image:
        cv2.imshow('img', img)
        cv2.waitKey(0)


def test_normal_image(engine, show_image=False):
    img = engine.get_normal_image()
    assert type(img) == np.ndarray
    assert img.ndim == 3
    assert img.shape[0] == 480
    assert img.shape[1] == 640
    assert img.shape[2] == 3
    if show_image:
        cv2.imshow('img', img)
        cv2.waitKey(0)


def test_depth_image(engine, show_image=False):
    img = engine.get_depth_image()
    assert type(img) == np.ndarray
    assert img.ndim == 2
    assert img.shape[0] == 480
    assert img.shape[1] == 640
    if show_image:
        img_scaled = img / 10.
        img_scaled[img_scaled > 1] = 0
        cv2.imshow('img', img_scaled)
        cv2.waitKey(0)


if __name__ == '__main__':
    engine = UnrealCVWrapper()
    show_image = True
    test_rgb_image(engine, show_image)
    test_normal_image(engine, show_image)
    test_depth_image(engine, show_image)
