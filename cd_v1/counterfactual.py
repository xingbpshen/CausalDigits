import pandas as pd
import numpy as np
import cv2


def draw_digit(img_size, digit, thickness, position, digit_size, rotation, thres=None):
    """
    Draws a digit with specified properties (with parameters relative to img_size)
    by rotating the digit first, then positioning it, and finally returning the image.

    Parameters:
        img_size (tuple): The size of the image (e.g., (64, 64)).
        digit (int): The digit to be drawn (0-9).
        thickness (int): The thickness of the digit
        position (tuple): The (x, y) normalized position of the center of the digit (0,0) to (1,1).
        digit_size (int): The size of the digit (1-10, discrete, relative to img_size).
        rotation (float): The clockwise rotation of the digit in degrees.

    Returns:
        np.ndarray: A 2D numpy array representing the image of the digit.
    """
    rotation = -rotation
    # Create a blank temporary canvas
    temp_canvas_size = (img_size[0] * 2, img_size[1] * 2)  # Larger to avoid cropping during rotation
    temp_canvas = np.zeros(temp_canvas_size, dtype=np.uint8)

    # Scale thickness relative to the image size
    # thickness_scaled = max(1, int(thickness / 10 * min(img_size)))
    thickness_scaled = thickness

    # Scale digit size relative to the image size
    font_scale = digit_size / 10 * (min(img_size) / 20.0)

    # Get the size of the text for positioning
    text_size = cv2.getTextSize(str(digit), cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_scaled)[0]
    center_x = temp_canvas_size[1] // 2 - text_size[0] // 2
    center_y = temp_canvas_size[0] // 2 + text_size[1] // 2

    # Draw the digit centered on the temporary canvas
    cv2.putText(temp_canvas, str(digit), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255),
                thickness_scaled)

    # Rotate the temporary canvas
    M = cv2.getRotationMatrix2D((temp_canvas_size[1] // 2, temp_canvas_size[0] // 2), rotation, 1)
    rotated_canvas = cv2.warpAffine(temp_canvas, M, temp_canvas_size, flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Create a final blank canvas
    final_canvas = np.zeros(img_size, dtype=np.uint8)

    # Calculate the position in pixel values
    pos_x = int(position[0] * img_size[1])
    pos_y = int(position[1] * img_size[0])

    # Overlay the rotated digit onto the final canvas at the specified position
    x_offset = pos_x - temp_canvas_size[1] // 2
    y_offset = pos_y - temp_canvas_size[0] // 2

    for i in range(temp_canvas_size[0]):
        for j in range(temp_canvas_size[1]):
            if 0 <= y_offset + i < img_size[0] and 0 <= x_offset + j < img_size[1]:
                if rotated_canvas[i, j] > 0:
                    final_canvas[y_offset + i, x_offset + j] = rotated_canvas[i, j]

    if thres is not None:
        final_canvas[final_canvas < thres] = 0
        final_canvas[final_canvas >= thres] = 255

    return final_canvas


def edit_digit(img, digit_color, has_bar):
    """
    Transform a 2D grayscale image (64, 64) into a 3-channel image with specified color settings.

    Parameters:
    - img: 2D numpy array of shape (64, 64), values are either 0 or 255.
    - digit_color: Binary, 0 means green, 1 means red.
    - has_bar: Binary, 0 means no bar, 1 means a blue bar is added at the top.
    """
    # Ensure img is 2D with values either 0 or 255
    assert img.shape == (64, 64), "Image must be 64x64"
    assert np.all(np.isin(img, [0, 255])), "Image must only contain values 0 or 255"

    # Create a 3-channel (RGB) image initialized to black
    img_color = np.zeros((64, 64, 3), dtype=np.uint8)

    # Define colors for the digit based on digit_color input
    if digit_color == 0:  # Green color
        img_color[img == 255] = [0, 255, 0]
    else:  # Red color
        img_color[img == 255] = [255, 0, 0]

    # Add a blue bar (width 6 pixels, full length) on top if has_bar is 1
    if has_bar == 1:
        img_color[:6, :] = [0, 0, 255]  # Blue color

    return img_color


def gt_ctf(original_img_ids: np.array, ctf_queries: list, path_to_attributes):
    # Example input:
    # original_img_ids = np.array([0, 1]) # ids of n original images, here n = 2
    # ctf_queries = [{"D": 8, "C": 1}, {"B": 0}] # intervention, len(ctf_queries) = n
    # path_to_attributes = "./CausalDigits/cd_v1/attributes.csv"
    # return an np array of shape (n, 64, 64, 3), n ctf images
    attributes_data = pd.read_csv(path_to_attributes)
    outputs = []
    # Find the row with id and convert it to a dictionary
    for i in range(len(original_img_ids)):
        id = original_img_ids[i]
        ctf_query = ctf_queries[i]
        row_dict = attributes_data[attributes_data['id'] == id].to_dict(orient='records')[0]
        # Define the keys you want to assign
        keys = ['D', 'C', 'B']
        # Assign values from ctf_query if they exist, otherwise use values from row_dict
        values = {key: ctf_query.get(key, row_dict.get(key)) for key in keys}
        # Unpack the values into variables D, C, and B
        D, C, B = int(values['D']), int(values['C']), int(values['B'])
        # Special case for B, if not in the query, recalculate B from the SCM with abducted exogenous noises
        if "B" not in ctf_query.keys():
            B = ((1 * (D >= 5) ^ int(row_dict["U1"])) or (C ^ int(row_dict["U2"]))) and int(row_dict["U3"])
        _img = draw_digit(img_size=(64, 64),
                          digit=int(D),
                          thickness=int(row_dict["Thickness"]),
                          position=(float(row_dict["Position0"]), float(row_dict["Position1"])),
                          digit_size=int(row_dict["DigitSize"]),
                          rotation=int(row_dict["Rotation"]),
                          thres=180)
        _img = edit_digit(_img, int(C), int(B))
        outputs.append(_img)
    outputs = np.stack(outputs, axis=0)

    return outputs
