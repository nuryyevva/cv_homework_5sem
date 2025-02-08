import cv2
import numpy as np

def find_width_of_wall(image):
    """
    Finds the width of a wall in an image, assuming the wall is a contiguous
    black region starting from the top-left corner.
    """

    # Iterate through the image, checking for a non-black pixel
    for i in range(image.shape[0]):
        if not np.all(image[:i, :i] == [0, 0, 0]):
            break

    # Return the width (i - 1) since the loop ends when a non-black pixel is found
    return i - 1

def find_width_of_path(image):
    """
    Determines the width of a path in an image, assuming the path is 
    represented by a specific color that appears consistently along the 
    first row of the image.
    """

    # Get unique colors and their counts in the first row
    unique_colors, color_counts = np.unique(image[0], return_counts=True)

    # Find the count of the path color
    path_width = color_counts[1] // 3  # Divide by 3 to account for RGB channels

    return path_width

def search_path(image, start, end, wall, path):
    """
    Finds a path through the maze using a breadth-first search (DFS) algorithm.
    """

    # Initialize path list, path index, and visited points set
    path_list = [[start]]
    path_index = 0
    visited_points = {start}

    # If start and end are the same, return the start point
    if start == end:
        return path_list[0]

    # Loop through each path in the path list
    while path_index < len(path_list):
        current_path = path_list[path_index]
        last_point = current_path[-1]

        # Check for valid next points from the last point in the current path
        next_points = check(image, last_point[0], last_point[1], wall, path)

        # If end is a next point, path found
        if end in next_points:
            current_path.append(end)
            return current_path

        # Add new paths to path list for each valid next point
        for next_point in next_points:
            if next_point not in visited_points:
                new_path = current_path[:]
                new_path.append(next_point)
                path_list.append(new_path)
                visited_points.add(next_point)

        # Move to the next path in the list
        path_index += 1

    # No path found
    return []


def paint_green(image, x, y, path):
    """Paints a green rectangle at the specified coordinates."""
    image = cv2.rectangle(image, (x, y), (x + path, y + path), (0, 200, 0), -1)
    return image


def check(image, x, y, wall, path):
    """
    Checks for valid next points from the current point.
    """

    next_points = []

    # Check down
    i = 0
    while y + i < len(image) - 1 and np.all(image[y + i, x] != [0, 0, 0]) and i < path + wall + 1:
        i += 1
    if i == path + wall + 1:
        next_points.append((x, y + i))

    # Check left
    i = 0
    while x - i > -1 and np.all(image[y, x - i] != [0, 0, 0]) and i < path + wall + 1:
        i += 1
    if i == path + wall + 1:
        next_points.append((x - i, y))

    # Check right
    i = 0
    while x + i < len(image) - 1 and np.all(image[y, x + i] != [0, 0, 0]) and i < path + wall + 1:
        i += 1
    if i == path + wall + 1:
        next_points.append((x + i, y))

    # Check up
    i = 0
    while y - i > -1 and np.all(image[y - i, x] != [0, 0, 0]) and i < path + wall + 1:
        i += 1
    if i == path + wall + 1:
        next_points.append((x, y - i))

    return next_points
    
def find_way_from_maze(image1: np.ndarray) -> tuple:
    """
    Finds a path through a maze represented by an image.
    """
    image = image1.copy()

    # Find wall and path widths
    wall_width = find_width_of_wall(image)
    path_width = find_width_of_path(image) - 1

    # Find start and end points
    start_row = 0
    while np.all(image[:1, :start_row] == [0, 0, 0]):
        start_row += 1
    start = (start_row - 1, wall_width)

    end_row = len(image) - 1
    end_col = 0
    while np.all(image[end_row:end_row + 1, :end_col] == [0, 0, 0]):
        end_col += 1
    end = (end_col - 1, end_row - wall_width - path_width)

    # Mark start and end points
    image = cv2.rectangle(image, (start[0], 0), (start[0] + path_width, wall_width - 1), (0, 200, 0), -1)
    image = cv2.rectangle(image, (end[0], end[1] + path_width + 1), (end[0] + path_width, end_row), (0, 200, 0), -1)

    # Find path using search algorithm
    path_coords = search_path(image, start, end, wall_width, path_width) 

    # Paint the path
    for row, col in path_coords:
        paint_green(image, row, col, path_width)

    # Extend the path to cover adjacent white cells
    for row in range(len(image)):
        for col in range(len(image)):
            if col + wall_width + 1 < len(image) and \
               np.all(image[row, col] == [0, 200, 0]) and \
               np.all(image[row, col + wall_width + 1] == [0, 200, 0]):
                if np.all(image[row, col + 1] == [255, 255, 255]):
                    image = cv2.rectangle(image, (col + 1, row), (col + wall_width, row + path_width), (0, 200, 0), -1)
            if col + wall_width + 1 < len(image) and \
               np.all(image[col, row] == [0, 200, 0]) and \
               np.all(image[col + wall_width + 1, row] == [0, 200, 0]):
                if np.all(image[col + 1, row] == [255, 255, 255]):
                    image = cv2.rectangle(image, (row, col + 1), (row + path_width, col + wall_width), (0, 200, 0), -1)

    # Get coordinates of the path
    X, Y = np.where(np.all(image == [0, 200, 0], axis=2))

    return X, Y
