import os
import random
import shutil
import sys
import numpy as np
import cv2
sys.path.append('../Flow generation/FlowFormerPlusPlus')  # add FlowFormerPlusPlus to path; IDE path config is required
import visualize_flow


"""Dataset root/output directories."""
root_dir_reds = r'E:\sdata\REDS\train\train_sharp'  # REDS dataset path
output_dir_reds = r"E:/data/Noflare_Reds"  # Select training images and copy to a new location (Windows)
root_dir_ldv3 = r'E:\sdata\LDV3\train_frames'  # LDV3 dataset path
flow_dir_reds = r'E:/data/Flow_Reds'  # REDS optical-flow output path (Windows)
flare_r_origin_data = r'E:\Video_Flare_Removal\data\Flare_mask\scattering_light\Flare7K-R\Compound_Flare'
flare_r_new_gamma = r'E:\Video_Flare_Removal\data\Flare_mask\scattering_light\Flare7K-R\Compound_Flare_gamma'
flare_r_new_sub = r'E:\Video_Flare_Removal\data\Flare_mask\scattering_light\Flare7K-R\Compound_Flare_subtract'
# Source folder path, training-set folder path, and test-set folder path
source_folder = r'E:\Video_Flare_Removal\data\WithFlare\autodl_data\with_scattering_and_reflection_light'  # Source folder
train_folder = r'E:\Video_Flare_Removal\data\WithFlare\autodl_data\trainset\lq'  # Training folder
test_folder = r'E:\Video_Flare_Removal\data\WithFlare\autodl_data\testset\lq'  # Test folder


"""Dataset subset sampling."""
def generate_frame_filenames(scene_dir, start_frame, original_data_name):
    """Generate filenames for a consecutive 8-frame clip."""
    if original_data_name == 'REDS':
        return [os.path.join(scene_dir, f'{start_frame + i:08d}.png') for i in range(8)]
    elif original_data_name == 'LDV3':
        print(start_frame)
        return [os.path.join(scene_dir, f'{start_frame + i}.png') for i in range(8)]


def create_folder_and_copy_frames(output_folder, filenames):
    """Create a new folder and copy frames from the original dataset."""
    os.makedirs(output_folder, exist_ok=True)
    for i, filename in enumerate(filenames):
        new_filename = os.path.join(output_folder, f'{i:03d}.png')
        # os.link(filename, new_filename)  # On Linux, link() can be used for hard links
        shutil.copy2(filename, new_filename)


def process_noflare_data(total_folder, scenes_list, root_dir, output_dir, original_data_name):
    """
    Generate a new dataset by sampling clips from an existing dataset.

    :param total_folder: Number of output folders (i.e., number of sampled scenes)
    :param scenes_list: List of all scenes in the original dataset
    :param root_dir: Root directory of the original dataset
    :param output_dir: Output directory
    :param original_data_name: Dataset name (e.g., 'REDS' or 'LDV3')
    """
    for folder_idx in range(total_folder):
        # Randomly pick a scene
        scene = random.choice(scenes_list)
        scene_dir = os.path.join(root_dir, scene)
        frames = []
        # List all frames in the scene; filename formats differ across datasets
        if original_data_name == 'REDS':
            frames = sorted([f for f in os.listdir(scene_dir) if f.endswith('.png')])
        elif original_data_name == 'LDV3':
            frames = sorted([f for f in os.listdir(scene_dir) if f.endswith('.png')], key=lambda f: int(f[:-4]))
        # Choose the start frame
        start_frame = random.randint(0, len(frames) - 8)
        # Generate filenames for a consecutive 8-frame clip
        filenames = generate_frame_filenames(scene_dir, int(frames[start_frame][:-4]), original_data_name)
        # Create output folder and copy frames
        output_folder = os.path.join(output_dir, f'{folder_idx:04d}')
        create_folder_and_copy_frames(output_folder, filenames)
        print(f'Processed {output_folder} finished')


def select_noflare_data():
    """
    Generate a new dataset extracted from the original dataset and split into scenes.
    :return:
    """
    os.makedirs(output_dir_reds, exist_ok=True)
    scenes_reds_list = [f for f in os.listdir(root_dir_reds) if os.path.isdir(os.path.join(root_dir_reds, f))]
    total_folders_for_reds = len(scenes_reds_list) * 8  


"""Optical-flow generation."""
def flow_generation(total_pic_folder, total_flow_folder):
    sub_dirs = os.listdir(total_pic_folder)
    for sub_dir in sub_dirs:
        print(f'{sub_dir} is processing...........................')
        source_pic_dir = os.path.join(total_pic_folder, sub_dir).replace("\\", "/")
        target_flow_dir = os.path.join(total_flow_folder, sub_dir).replace("\\", "/")
        os.makedirs(target_flow_dir, exist_ok=True)
        visualize_flow.generate_flow(source_pic_dir, target_flow_dir, 'seq')


"""Flare-R preprocessing."""
def gamma_preprocess(image, gamma):
    # Gamma correction
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def subtract_preprocess(image):
    # Subtraction-based adjustment
    avg_brightness = np.mean(image)
    adjusted_image = np.clip(image - avg_brightness, 0, 255).astype(np.uint8)
    return adjusted_image

def flare_r_preprocess(input_folder_path, output_folder_path, method):
    img_list = os.listdir(input_folder_path)
    for i in range(len(img_list)):
        img = cv2.imread(os.path.join(input_folder_path, img_list[i]))
        if method == 'gamma':
            img = gamma_preprocess(img, 1/2.2)
        elif method == 'subtract':
            img = subtract_preprocess(img)
        cv2.imwrite(os.path.join(output_folder_path, img_list[i]), img)
        print(f'{img_list[i]} is processed')


"""Dataset split."""
def split_dataset(source_folder, train_folder, test_folder, train_ratio=0.8):
    """
    Split subfolders in the source directory into train/test sets and copy images to the new directories.

    :param source_folder: Source folder path
    :param train_folder: Training folder path
    :param test_folder: Test folder path
    :param train_ratio: Train split ratio (default: 0.8)
    """
    # Create train/test directories
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Collect all subfolders
    subfolders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]
    random.seed(42)
    random.shuffle(subfolders)  # Shuffle subfolder order

    # Compute train/test counts
    train_count = int(len(subfolders) * train_ratio)
    train_subfolders = subfolders[:train_count]
    test_subfolders = subfolders[train_count:]

    # Copy train subfolders
    for subfolder in train_subfolders:
        src_subfolder = os.path.join(source_folder, subfolder)
        dst_subfolder = os.path.join(train_folder, subfolder)
        os.makedirs(dst_subfolder, exist_ok=True)
        for img_file in os.listdir(src_subfolder):
            src_img = os.path.join(src_subfolder, img_file)
            dst_img = os.path.join(dst_subfolder, img_file)
            shutil.copy2(src_img, dst_img)

    # Copy test subfolders
    for subfolder in test_subfolders:
        src_subfolder = os.path.join(source_folder, subfolder)
        dst_subfolder = os.path.join(test_folder, subfolder)
        os.makedirs(dst_subfolder, exist_ok=True)
        for img_file in os.listdir(src_subfolder):
            src_img = os.path.join(src_subfolder, img_file)
            dst_img = os.path.join(dst_subfolder, img_file)
            shutil.copy2(src_img, dst_img)


if __name__ == '__main__':
    # select_noflare_data()
    # flow_generation(output_dir_ldv3, flow_dir_ldv3)
    # flare_r_preprocess(flare_r_origin_data, flare_r_new_gamma, 'gamma')
    split_dataset(source_folder, train_folder, test_folder, train_ratio=0.8)
