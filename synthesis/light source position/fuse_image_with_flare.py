import cv2
import os
import numpy as np
import random


"""Paths for various assets."""
# Light-source position folder
input_folder_light_position_reds = r'E:\Video_Flare_Removal\data\Sourcelight_position_new'
# Physically-generated scattering flare templates
input_folder_physics_lightsource = r'E:\Video_Flare_Removal\data\Flare_mask_new\scattering_light\Synthesis\cleanlight'
# Physically-generated reflective flare templates
input_folder_physics_reflective_light_reds = r'E:\Video_Flare_Removal\data\Flare_mask_new\reflective_light'
# Original (no-flare) image folder
input_folder_reds_origin = r'E:\Video_Flare_Removal\data\Noflare_Reds'
# Output: images with scattering flare only
output_folder_scattering_light_reds = r'E:\Video_Flare_Removal\data\WithFlare\with_scattering_light'
# Output: images with light source only
output_folder_lightsource_reds = r'E:\Video_Flare_Removal\data\WithFlare\only_with_sourcelight'
# Output: images with light source + reflective flare
output_folder_all_light_reds = r'E:\Video_Flare_Removal\data\WithFlare\with_scattering_and_reflection_light'


"""Helper functions."""
def overlay_image(image1, image2, position, image1_scale):
    """
    Overlay image1 on image2.

    :param image1: Simulated flare image array
    :param image2: Scene/image array
    :param position: Composite position (center)
    :param image1_scale: Flare scale factor (controls flare size)
    :return: output_image (composited result)
    """
    image1 = cv2.resize(image1, (int(image1.shape[1] * image1_scale), int(image1.shape[0] * image1_scale)))
    height, width = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    origin_mask = np.zeros((8000, 8000, 1), np.uint8)  # Composite canvas; usually > 2*(image1*scale + image2) is safe

    for i in range(1, height):
        for j in range(1, width):
            origin_mask[i + int(4000 - height / 2)][j + int(4000 - width / 2)] = image1[i][j]
    noise = origin_mask[int(4000 - position[0]):int(4000 - position[0] + height2), int(4000 - position[1]):int(4000 - position[1] + width2)]
    noise = cv2.resize(noise, (width2, height2))
    if noise.ndim != 3:
        noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
        if noise.shape[2] != 4:
            noise = cv2.cvtColor(noise, cv2.COLOR_BGR2BGRA)
    if image2.shape[2] != 4:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2BGRA)
    image_mixed = cv2.addWeighted(image2, 1, noise, 1, 0)
    return image_mixed


def overlay_image_avoid_overexposure(image1, image2, position, image1_scale):
    """
    Overlay image1 on image2 (variant intended to reduce overexposure).

    :param image1: Simulated flare image array
    :param image2: Scene/image array
    :param position: Composite position (center)
    :param image1_scale: Flare scale factor (controls flare size)
    :return: output_image (composited result)
    """
    image1 = cv2.resize(image1, (int(image1.shape[1] * image1_scale), int(image1.shape[0] * image1_scale)))
    height, width = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    origin_mask = np.zeros((8000, 8000, 3), np.uint8)  # Composite canvas; usually > 2*(image1*scale + image2) is safe

    for i in range(1, height):
        for j in range(1, width):
            origin_mask[i + int(4000 - height / 2)][j + int(4000 - width / 2)] = image1[i][j]
    noise = origin_mask[int(4000 - position[0]):int(4000 - position[0] + height2), int(4000 - position[1]):int(4000 - position[1] + width2)]
    noise = cv2.resize(noise, (width2, height2))
    if noise.ndim != 3:
        noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
    if noise.shape[2] != 4:
        noise = cv2.cvtColor(noise, cv2.COLOR_BGR2BGRA)
    if image2.shape[2] != 4:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2BGRA)
    mix_with_alpha = fuse_scene_light(image2, noise)
    image_mixed = cv2.addWeighted(image2, 1, noise, 1, 0)
    return image_mixed


def overlay_image_with_alpha_channel(background, foreground):
    """
    Blend using the alpha channel.

    Note: unlike overlay_image(), this function uses the alpha channel.

    :param background: Background image (default: 3 channels)
    :param foreground: Foreground image (default: 4 channels)
    :return: Blended image (default: 3 channels)
    """
    foreground = foreground.astype(float)
    background = background.astype(float)
    # 分离Alpha通道
    alpha_channel = foreground[:, :, 3:] / 255.0
    foreground = foreground[:, :, :3]
    # Alpha混合
    out_image = (1.0 - alpha_channel) * background + alpha_channel * foreground
    return np.clip(out_image, 0, 255).astype(np.uint8)


def gamma_inverse(image, gamma):
    """
    Apply gamma correction.

    :param image: Input image
    :param gamma: Gamma factor
    :return:
    """
    # Convert to float
    image = image.astype('float32') / 255.0
    # Gamma correction
    image = np.power(image, gamma)
    # Convert back to uint8 in [0, 255]
    image = (image * 255).astype('uint8')
    return image


def fuse_scene_light(scene, flare):
    I = flare[:, :, 0] + flare[:, :, 1] + flare[:, :, 2]
    I = I / 3
    a = np.random.random() * 4 + 3
    weight = 1 / (1 + np.e ** (-a * (I - 0.5)))
    weight = weight - np.min(weight)
    weight = weight / np.max(weight)
    a1 = (scene[:, :, 0] * (1 - weight) + flare[:, :, 0] * weight)
    a2 = (scene[:, :, 1] * (1 - weight) + flare[:, :, 1] * weight)
    a3 = (scene[:, :, 2] * (1 - weight) + flare[:, :, 2] * weight)
    result = np.stack([a1, a2, a3], axis=-1)
    return np.clip(result, 0.0, 1.0)


"""Main compositing functions - reflective flare."""
def blend_images(folder1, folder2, output_folder):
    """
    Iterate two folders, blend images with the same filenames, and save results.

    :param folder1: First image folder path
    :param folder2: Second image folder path
    :param output_folder: Output folder path
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # List filenames in the first folder
    images1 = [f for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))]
    for filename in images1:
        # Build full paths
        img_path1 = os.path.join(folder1, filename)
        img_path2 = os.path.join(folder2, filename)
        # Only proceed if the second folder contains the same filename
        if os.path.isfile(img_path2):
            # Read both images
            img1 = cv2.imread(img_path1)
            img2 = cv2.imread(img_path2)
            img1 = gamma_inverse(img1, 2.2)  # Reflective flares were not gamma-encoded initially; apply gamma 2.2 before blending
            # Image blending: simple example; real usage may require more advanced blending
            blended_img = None
            try:
                alpha = random.uniform(0.8, 1.0)
                blended_img = cv2.addWeighted(img1, 1, img2, alpha, 0)  # alpha blending
                blended_img = gamma_inverse(blended_img, 1 / 2.2)
                # Save blended image
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, blended_img)
                print(f"Blended image {filename} is done")
            except Exception as e:
                ValueError(f"Error blending images: {e}")
        else:
            print(f"Image {filename} not found in the second folder.")


def fuse_rf_sc_light_pic(folder_sc_light_pic, folder_rf_light_mask, output_folder):
    """
    Combine two types of flares.

    :param folder_sc_light_pic: Root folder containing scattering-flare images
    :param folder_rf_light_mask: Root folder containing reflective-flare templates
    :param output_folder: Output folder containing both flare effects
    :return: None
    """
    print('****combine scattering flare image and reflective flare****')
    scene_dir_list = os.listdir(folder_sc_light_pic)
    for scene_dir in scene_dir_list:
        print(f'scene{scene_dir} is start')
        per_scene_sc_light_path = os.path.join(folder_sc_light_pic, scene_dir).replace('\\', '/')
        per_scene_rf_light_path = os.path.join(folder_rf_light_mask, scene_dir).replace('\\', '/')
        output_folder_path = os.path.join(output_folder, scene_dir)
        blend_images(per_scene_sc_light_path, per_scene_rf_light_path, output_folder_path)
        print(f'scene{scene_dir} is end')


"""Main compositing functions - scattering flare and light source."""
def generate_light_source_real_simulate(input_image, light_source_center, streaks_dir, scale):
    """
    Generate an image with a simulated flare/light source.

    :param input_image: Input image
    :param light_source_center: Light source center
    :param streaks_dir: Flare/streak image path
    :param scale: Flare image scaling factor
    :return: output_image (image with simulated light source)
    """
    light_source_pic = cv2.imread(f'{streaks_dir}')
    light_source_pic = gamma_inverse(light_source_pic, 2.2)
    output_image = overlay_image_avoid_overexposure(light_source_pic, input_image, light_source_center, scale)
    output_image = gamma_inverse(output_image, 1 / 2.2)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGRA2BGR)
    return output_image


def mk_sclight_lightsource_list(physics_sc_light_folder, physics_lightsource_folder):
    # sc_light_list = [os.path.join(physics_sc_light_folder, f).replace('\\', '/') for f in os.listdir(physics_sc_light_folder)] + [os.path.join(flare7kr_sc_light_folder, f).replace('\\', '/') for f in os.listdir(flare7kr_sc_light_folder)]
    # lightsource_list = [os.path.join(physics_lightsource_folder, f).replace('\\', '/') for f in os.listdir(physics_lightsource_folder)] + [os.path.join(flare7kr_source_folder, f).replace('\\', '/') for f in os.listdir(flare7kr_source_folder)]
    sc_light_list = [os.path.join(physics_sc_light_folder, f).replace('\\', '/') for f in os.listdir(physics_sc_light_folder)]
    lightsource_list = [os.path.join(physics_lightsource_folder, f).replace('\\', '/') for f in os.listdir(physics_lightsource_folder)]
    return sc_light_list, lightsource_list


def fuse_pic_with_sc_light_and_light_source(origin_pics_path, path_lightsource_position, output_sclight_pic_folder, output_lightsource_pic_folder):
    """
    Combine scattering flare and light source.

    :param origin_pics_path: Path to original images
    :param path_lightsource_position: Path to light-source position files
    :param output_sclight_pic_folder: Output folder for images with scattering flare
    :param output_lightsource_pic_folder: Output folder for images with light source
    :return: None
    """
    print('****combine scattering light and light source****')
    # Build path lists for scattering-flare and light-source masks
    sclight_list, lightsource_list = mk_sclight_lightsource_list(input_folder_physics_scattering_light, input_folder_physics_lightsource)
    # Read per-scene coordinates and build relevant path indices
    lightsource_position_list = os.listdir(path_lightsource_position)
    for i in range(len(lightsource_position_list)):
        lightsource_pt_path = os.path.join(path_lightsource_position, lightsource_position_list[i]).replace('\\', '/')
        origin_scene_path = os.path.join(origin_pics_path, f'{i:04}').replace('\\', '/')
        output_sc_scene_folder = os.path.join(output_sclight_pic_folder, f'{i:04}').replace('\\', '/')
        output_lightsource_scene_folder = os.path.join(output_lightsource_pic_folder, f'{i:04}').replace('\\', '/')
        os.makedirs(output_sc_scene_folder, exist_ok=True)
        os.makedirs(output_lightsource_scene_folder, exist_ok=True)
        coords = None
        if lightsource_pt_path.endswith(".txt"):
            coords = np.loadtxt(lightsource_pt_path)
        else:
            ValueError("Please check the lightsource position file")
        # Choose a flare/light-source pair
        random_index = random.randrange(0, len(sclight_list))
        sclight_mask_path = sclight_list[random_index]
        lightsource_mask_path = lightsource_list[random_index]
        lightsource_scale = 1
        lightsource_scale = round(random.uniform(1.5, 3.0), 1)
        print(f"scene {i:04} is start fuse scattering light and light source")
        # Process frames within a single scene
        for j in range(coords.shape[0]):
            origin_pic_path = os.path.join(origin_scene_path, f'{j:03}.png').replace('\\', '/')
            origin_image = cv2.imread(origin_pic_path)
            origin_image = gamma_inverse(origin_image, 2.2)
            lightsource_center = (coords[j, 0], coords[j, 1])
            # Compose two variants and save
            pic_with_sclight = generate_light_source_real_simulate(origin_image, lightsource_center, sclight_mask_path, lightsource_scale)
            pic_with_light_source = generate_light_source_real_simulate(origin_image, lightsource_center, lightsource_mask_path, lightsource_scale)
            cv2.imwrite(os.path.join(output_sc_scene_folder, f'{j:03}.png').replace('\\', '/'), pic_with_sclight)
            cv2.imwrite(os.path.join(output_lightsource_scene_folder, f'{j:03}.png').replace('\\', '/'), pic_with_light_source)
            print(f"pic {j:03} is done")
        print(f"****************scene {i:04} is done************************")


if __name__ == "__main__":
    # Compose light source and scattering flare
    fuse_pic_with_sc_light_and_light_source(input_folder_reds_origin, input_folder_light_position_reds, output_folder_scattering_light_reds, output_folder_lightsource_reds)
    # Combine scattering and reflective flares
    fuse_rf_sc_light_pic(output_folder_scattering_light_reds, input_folder_physics_reflective_light_reds, output_folder_all_light_reds)
