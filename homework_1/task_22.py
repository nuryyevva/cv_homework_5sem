import cv2
import numpy as np

def find_road_number(image: np.ndarray) -> int:
    """
    Determines the number of the road the car should be on based on 
    analyzing the image for yellow lines, red obstacles, and the blue car.
    """
    # Convert image to HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Create masks for red obstacles, yellow lines, and blue car
    lower_red_mask = cv2.inRange(image_hsv, np.array([0, 100, 20]), np.array([10, 255, 255]))
    upper_red_mask = cv2.inRange(image_hsv, np.array([160, 100, 20]), np.array([179, 255, 255]))
    obstacles_mask = lower_red_mask + upper_red_mask

    yellow_line_mask = cv2.inRange(image_hsv, np.array([25, 50, 70]), np.array([35, 255, 255]))
    blue_car_mask = cv2.inRange(image_hsv, np.array([90, 50, 70]), np.array([128, 255, 255]))

    # Count the number of roads
    road_count = 0
    yellow_line_size = 0
    i = 0
    while i < len(yellow_line_mask[0]):
        if road_count == 0 and yellow_line_mask[0, i] == 255 and yellow_line_mask[0, i + 1] == 0:
            road_count += 1
            yellow_line_size = i + 1
        if yellow_line_mask[0, i] == 255 and yellow_line_mask[0, i - 1] == 0:
            i += yellow_line_size
            road_count += 1
        else:
            i += 1

    # Calculate road size and step
    road_size = (len(image_hsv[0]) - road_count * yellow_line_size) // (road_count - 1)
    step = yellow_line_size + road_size

    # Find the empty road index and car index
    empty_road_index = -1
    car_index = -1
    i = 0
    while i < (len(image_hsv[0]) - step):
        if np.all(obstacles_mask[0:, i:i + step] == 0):
            empty_road_index = i // step
        if np.all(blue_car_mask[0:, i:i + step] == 0) == False:
            car_index = i // step
        i += step

    # Determine if road change is needed
    if empty_road_index == car_index:
        print("No need to change the road ^-^")
        return -1

    # Return the index of the empty road
    return empty_road_index
    