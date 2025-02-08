import cv2
import numpy as np


def rotate(image, point: tuple, angle: float) -> np.ndarray:
    """
    Повернуть изображение по часовой стрелке на угол от 0 до 360 градусов и преобразовать размер изображения.

    :param image: исходное изображение
    :param point: значение точки (x, y), вокруг которой повернуть изображение
    :param angle: угол поворота
    :return: повернутное изображение
    """
    rows, cols, _ = image.shape

    # Calculate rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(point, angle, scale=1.0)

    # Determine the shift required for the rotated image
    corners = np.array([[0, 0], [cols, 0], [0, rows], [cols, rows]], dtype=np.float32)
    transformed_corners = cv2.transform(corners.reshape(-1, 1, 2), rotation_matrix)
    min_x, min_y = np.min(transformed_corners, axis=0)[0]
    max_x, max_y = np.max(transformed_corners, axis=0)[0]
    shift_x = abs(int(min_x))
    shift_y = abs(int(min_y))

    # Adjust rotation matrix for the shift
    rotation_matrix[0, 2] += shift_x
    rotation_matrix[1, 2] += shift_y

    # Apply the rotation and resize the image
    return cv2.warpAffine(image.copy(), rotation_matrix, (shift_x + int(max_x) + 1, shift_y + int(max_y) + 1))
    