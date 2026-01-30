%% Batch generate comparison flare images
% Generates 8 images:
% Group A (center crop): 4 images
%   - Img1: circular aperture + streak contamination (angle 1)
%   - Img2: circular aperture + streak contamination (angle 2)
%   - Img3: hexagonal aperture + streak contamination (angle 3)
%   - Img4: hexagonal aperture + streak contamination (angle 4)
% Group B (random crop): 4 images (same configurations as Group A)
%
clear
close all

%% Camera parameters
lambda = 550e-9;  % Wavelength
f = 2.2e-3;       % Focal length
delta = 1e-6;     % Pixel pitch
l = 3e-3;         % Sensor size
res = l / delta;  % Simulation resolution

%% Fourier-domain parameters
lf = lambda * f / delta;
df = 1e-3;
rf_norm = df / 2 / lf;

%% Spectral response sampling
num_wavelengths = 73;
wavelengths = linspace(380, 740, num_wavelengths) * 1e-9;

%% Output parameters
out_crop = 600;
defocus_crop = 2000;

%% Create output directories
out_dir_base = '../res_comparison/';
mkdir(out_dir_base);
out_dir_center = [out_dir_base, 'group_A_center/'];
out_dir_random = [out_dir_base, 'group_B_random/'];
out_dir_apertures = [out_dir_base, 'apertures/'];
mkdir(out_dir_center);
mkdir(out_dir_random);
mkdir(out_dir_apertures);

%% Define streak angles (radians) - four distinct angles
streak_angles = [pi/6, pi/3, pi/4, 2*pi/3];  % 30°, 60°, 45°, 120°

%% Define configurations
% Format: {aperture_type, streak_angle_index, output_name}
configs = {
    'circle', 1, 'img1_circle_streak30deg';
    'circle', 2, 'img2_circle_streak60deg';
    'poly_6', 3, 'img3_hex_streak45deg';
    'poly_6', 4, 'img4_hex_streak120deg'
};

%% Random spectral response (shared across all images for comparability)
wl_to_rgb = RandomSpectralResponse(wavelengths).';

%% Main loop: generate 8 images
fprintf('Starting comparison image generation...\n');
fprintf('=========================================\n');

for config_idx = 1:size(configs, 1)
    aperture_type = configs{config_idx, 1};
    angle_idx = configs{config_idx, 2};
    base_name = configs{config_idx, 3};
    streak_angle = streak_angles(angle_idx);
    
    fprintf('\nProcessing config %d/4: %s\n', config_idx, base_name);
    fprintf('  Aperture type: %s\n', aperture_type);
    fprintf('  Streak angle: %.1f deg\n', rad2deg(streak_angle));
    
    %% Compute defocus phase and aperture mask for this configuration
    [defocus_phase, aperture_mask] = GetDefocusPhase(res, rf_norm, aperture_type);
    
    %% Generate streak-contaminated aperture
    aperture = RandomDirtyAperture(aperture_mask, 'streak', streak_angle);
    
    %% Save aperture image
    aperture_filename = sprintf('aperture_%s.png', base_name);
    imwrite(aperture, [out_dir_apertures, aperture_filename]);
    fprintf('  Saved aperture: %s\n', aperture_filename);
    
    %% Random defocus
    defocus = randn * 5 + randn * 3;
    
    %% Compute PSF
    psf_rgb = GetPsf(aperture, defocus_phase * defocus, ...
                     wavelengths ./ lambda, wl_to_rgb, defocus_crop);
    
    %% Camera model parameters (for distortion)
    focal_length_px = f / delta * [1, 1];
    sensor_crop = [1200, 1200];
    principal_point = sensor_crop / 2;
    radial_distortion = [randn * 0.8, 0];
    camera_params = cameraIntrinsics( ...
        focal_length_px, principal_point, sensor_crop, ...
        'RadialDistortion', radial_distortion);
    
    %% Group A image (center crop)
    fprintf('  Generating Group A (center crop)...\n');
    
    % Center crop
    psf_cropped_center = CropCenter(psf_rgb, sensor_crop);
    psf_distorted_center = undistortImage(psf_cropped_center, camera_params);
    
    % Post-processing
    psf_ds_center = imresize(psf_distorted_center, 0.5, 'box');
    psf_out_center = EqualizeChannels(CropCenter(psf_ds_center, out_crop));
    psf_gamma_center = abs(psf_out_center .^ (1/2.2));
    psf_gamma_center = min(psf_gamma_center, 2^16 - 1);
    psf_u16_center = uint16(psf_gamma_center);
    
    % Save Group A image
    output_filename_center = sprintf('%s_center.png', base_name);
    imwrite(psf_u16_center, [out_dir_center, output_filename_center]);
    fprintf('    Saved: %s\n', output_filename_center);
    
    %% Group B image (random crop)
    fprintf('  Generating Group B (random crop)...\n');
    
    % True random crop
    window_random = randomCropWindow2d(size(psf_rgb), sensor_crop);
    psf_cropped_random = imcrop(psf_rgb, window_random);
    psf_distorted_random = undistortImage(psf_cropped_random, camera_params);
    
    % Post-processing
    psf_ds_random = imresize(psf_distorted_random, 0.5, 'box');
    psf_out_random = EqualizeChannels(CropCenter(psf_ds_random, out_crop));
    psf_gamma_random = abs(psf_out_random .^ (1/2.2));
    psf_gamma_random = min(psf_gamma_random, 2^16 - 1);
    psf_u16_random = uint16(psf_gamma_random);
    
    % Save Group B image
    output_filename_random = sprintf('%s_random.png', base_name);
    imwrite(psf_u16_random, [out_dir_random, output_filename_random]);
    fprintf('    Saved: %s\n', output_filename_random);
end

fprintf('\n=========================================\n');
fprintf('All images generated.\n');
fprintf('Output directory: %s\n', out_dir_base);
fprintf('  - Group A (center crop): %s\n', out_dir_center);
fprintf('  - Group B (random crop): %s\n', out_dir_random);
fprintf('  - Aperture images: %s\n', out_dir_apertures);
fprintf('=========================================\n');

%% Optional: preview the generated images
figure('Name', 'Comparison Preview', 'Position', [100, 100, 1400, 800]);

for i = 1:4
    % Show Group A (center crop)
    subplot(2, 4, i);
    img_center = imread([out_dir_center, configs{i, 3}, '_center.png']);
    imshow(img_center, []);
    title(sprintf('A%d: %s (center)', i, configs{i, 3}), 'Interpreter', 'none');
    
    % Show Group B (random crop)
    subplot(2, 4, i+4);
    img_random = imread([out_dir_random, configs{i, 3}, '_random.png']);
    imshow(img_random, []);
    title(sprintf('B%d: %s (random)', i, configs{i, 3}), 'Interpreter', 'none');
end

fprintf('\nPreview window opened.\n');
