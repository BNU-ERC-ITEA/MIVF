import cv2
import os
import sys
import numpy as np
import os.path as osp
from glob import glob
import random
import math

# Path configuration
noflare_data_reds = r'E:\Video_Flare_Removal\data\Noflare_Reds'
noflare_data_ldv3 = r'E:\Video_Flare_Removal\data\Noflare_Ldv3'
flow_dir_reds = r'E:\Video_Flare_Removal\data\Flow_Reds'
flow_dir_ldv3 = r'E:\Video_Flare_Removal\data\Flow_Ldv3'
light_position_doc_reds = r'E:\Video_Flare_Removal\data\Sourcelight_position_new'


def random_start_point(image):
    """
    Randomly sample an initial light-source point.

    To reduce the chance that the point moves out of bounds, sample within the
    central 2/3 of the image, i.e. from 1/6 to 5/6 along each axis.

    :param image: Input image (used for its spatial size).
    :return: Random initial (row, col) position of the light source.
    """
    height, width = image.shape[:2]
    start_point = (
        random.randint(int(height / 6), int(5 * height / 6)), random.randint(int(width / 6), int(5 * width / 6)))
    return start_point


def random_start_point_ring(image):
    """
    Randomly sample a point from an outer ring-like region around the image.

    :param image: Input image (used for its spatial size).
    :return: Random (row, col) position in the outer ring-like region.
    """
    height, width = image.shape[:2]

    # Randomly pick one of the edge regions
    region = random.choice([
        (random.randint(int(height / 5), int(2 * height / 5) - 1), random.randint(int(width / 2 - 3 * height / 10), int(width / 2 + 3 * height / 10) - 1)),
        (random.randint(int(height / 5), int(4 * height / 5) - 1), random.randint(int(width / 2 - 3 * height / 10), int(width / 2 - 1 * height / 10))),
        (random.randint(int(3 * height / 5), int(4 * height / 5) - 1), random.randint(int(width / 2 - 3 * height / 10), int(width / 2 + 3 * height / 10 - 1))),
        (random.randint(int(height / 5), int(4 * height / 5) - 1), random.randint(int(width / 2 + 1 * height / 10), int(width / 2 + 3 * height / 10) - 1)),
    ])

    return region


def load_flow(x, y, i, flow_dir):
    """
    Load optical flow and query flow at a given pixel.

    :param x: x position
    :param y: y position
    :param i: Frame index.
    :param flow_dir: Optical-flow file path.
    :return: (flow_horizontal, flow_vertical) at (x, y).
    """
    flow = np.load(f"{flow_dir}")
    (flow_horizontal, flow_vertical) = flow[x][y]
    return [flow_horizontal, flow_vertical]


def generate_lightsource(noflare_pic_total_dir, flow_total_dir, light_pt_save_folder):
    """
    Generate a moving light-source track and save positions to a text file.

    The light-source position is updated frame-by-frame using the optical flow,
    and the per-frame positions are appended to a scene-specific .txt file.

    :param noflare_pic_total_dir: Directory containing the original (no-flare) frames.
    :param flow_total_dir: Directory containing optical flow for each scene.
    :param light_pt_save_folder: Output directory for saved light-source positions.
    :return:
    """
    scene_example = os.listdir(noflare_pic_total_dir)[0]
    dirname = f'{noflare_pic_total_dir}/{scene_example}'.replace('\\', '/')
    image_list = [os.path.join(dirname, x).replace('\\', '/') for x in os.listdir(dirname)]
    image_list.sort(key=lambda x: int(x.split('.')[-2].split('/')[-1]))
    image_example = cv2.imread(image_list[0])

    os.makedirs(light_pt_save_folder, exist_ok=True)
    for scene in os.listdir(noflare_pic_total_dir):
        print(f"=========>processing scene:{scene}=========>")
        # Process each scene

        # Initialize light-source position
        random_1st_pt = random_start_point_ring(image_example)
        print(f'original point:{random_1st_pt[0]} {random_1st_pt[1]}')
        position_lightsource = {0: random_1st_pt[0], 1: random_1st_pt[1]}  # Use a dict for the initial point
        # Prepare output file for light-source positions updated by optical flow
        light_positions_file = f'{light_pt_save_folder}/{scene}.txt'.replace('\\', '/')
        with open(light_positions_file, 'w') as file:
            file.write(f"{position_lightsource[0]} {position_lightsource[1]}\n")  # Write initial position
        # Iterate over frames in the scene
        for i in range(len(image_list) - 1):
            print(f"=========>processing image:{i}=========>")
            input_image = cv2.imread('{}'.format(image_list[i]))
            height, width = input_image.shape[:2]
            delta_position = load_flow(position_lightsource[0], position_lightsource[1], i,
                                       f'{flow_total_dir}/{scene}/{i:03}.png.npy'.replace('\\', '/'))
            position_lightsource = [delta_position[1] + position_lightsource[0],
                                    delta_position[0] + position_lightsource[1]]
            position_lightsource[0] = round(position_lightsource[0])  # Round flow_x
            position_lightsource[1] = round(position_lightsource[1])  # Round flow_y
            if position_lightsource[0] >= height:  # Clamp to image bounds
                position_lightsource[0] = height - 1
            if position_lightsource[1] >= width:
                position_lightsource[1] = width - 1
            if position_lightsource[0] <= 0:
                position_lightsource[0] = 1
            if position_lightsource[1] <= 0:
                position_lightsource[1] = 1
            # Append updated light-source position
            with open(light_positions_file, 'a') as file:
                file.write(f"{position_lightsource[0]} {position_lightsource[1]}\n")  # Append per frame


if __name__ == '__main__':
    generate_lightsource(noflare_data_reds, flow_dir_reds, light_position_doc_reds)
