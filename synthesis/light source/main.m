%  // clang-format off
clear
close all

%% Typical parameters for a smartphone camera.
% Nominal wavelength (m).
lambda = 550e-9;
% Focal length (m).
f = 2.2e-3;
% Pixel pitch on the sensor (m).
delta = 1e-6;
% Sensor size (width & height, m).
l = 3e-3;
% Simulation resolution, in both spatial and frequency domains.
res = l / delta;
% Aperture shape: 'poly_6' (hexagon) or 'circle'.
aperture_type = 'circle';

%% Compute defocus phase shift and aperture mask in the Fourier domain.
% Frequency range (extent) of the Fourier transform (m ^ -1).
lf = lambda * f / delta;
% Diameter of the circular low-pass filter on the Fourier plane.
df = 1e-3;
% Low-pass radius, normalized by simulation resolution.
rf_norm = df / 2 / lf;
[defocus_phase, aperture_mask] = GetDefocusPhase(res, rf_norm, aperture_type); % Fourier-domain defocus phase and aperture mask
[defocus_phase_purelight, aperture_mask_purelight] = GetDefocusPhase(res, rf_norm, aperture_type); % Same, for the clean (no-contamination) branch

%% Wavelengths at which the spectral response is sampled.
num_wavelengths = 73; % Number of wavelength samples for spectral response
wavelengths = linspace(380, 740, num_wavelengths) * 1e-9;

%% Create output directories.
% Create directories for contaminated apertures and corresponding flare PSFs.
if strcmp(aperture_type, 'circle')
    out_dir = '../res_paper/source_light_pic/circle/';
    mkdir(out_dir);
    aperture_dir = '../res_paper/apertures/circle/';
    mkdir(aperture_dir);
elseif strcmp(aperture_type, 'poly_6')
    out_dir = '../res_paper/source_light_pic/poly_6/';
    mkdir(out_dir);
    aperture_dir = '../res_paper/apertures/poly_6/';
    mkdir(aperture_dir);
else
    error('Invalid aperture shape. Choose ''circle'' or ''poly_6''.');
end
% Create directory for the corresponding clean point-light PSFs.
out_dir_purelight = '../res_paper/none/';
mkdir(out_dir_purelight);

out_crop = 600;

%% Generate the PSFs.
for tt = :42 % Generate multiple PSFs with different random apertures
  aperture = RandomDirtyAperture(aperture_mask, 'dirty'); % Random contaminated aperture (source_t: 'dirty' / 'none')
  aperture_purelight = RandomDirtyAperture(aperture_mask_purelight, 'none');
  imwrite(aperture, strcat(aperture_dir, sprintf('%03d.png',tt - 1)));

  %% Random RGB spectral response.
  wl_to_rgb = RandomSpectralResponse(wavelengths).';

  for ii = 1:2
    %% Random defocus.
    defocus_crop = 2000;
    defocus = randn * 5 + randn * 3;
    psf_rgb = GetPsf(aperture, defocus_phase * defocus, ...
                     wavelengths ./ lambda, wl_to_rgb, defocus_crop);
    psf_rgb_purelight = GetPsf(aperture_purelight, defocus_phase_purelight * defocus, ...
                     wavelengths ./ lambda, wl_to_rgb, defocus_crop);

    for kk = 1:2
      %% Randomly crop and distort the PSF.
      focal_length_px = f / delta * [1, 1];
      sensor_crop = [1200, 1200];
      principal_point = sensor_crop / 2;
      radial_distortion = [randn * 0.8, 0];
      camera_params = cameraIntrinsics( ...
          focal_length_px, principal_point, sensor_crop, ...
          'RadialDistortion', radial_distortion);
      psf_cropped = CropRandom(psf_rgb, sensor_crop);
      psf_cropped_purelight = CropRandom(psf_rgb_purelight, sensor_crop);
      psf_distorted = undistortImage(psf_cropped, camera_params);
      psf_distorted_purelight = undistortImage(psf_cropped_purelight, camera_params);

      %% Apply global tone curve (gamma) and write to disk.
      psf_ds = imresize(psf_distorted, 0.5, 'box');
      psf_ds_purelight = imresize(psf_distorted_purelight, 0.5, 'box');
      psf_out = EqualizeChannels(CropCenter(psf_ds, out_crop));
      % psf_out = CropCenter(psf_ds, out_crop);
      psf_out_purelight = EqualizeChannels(CropCenter(psf_ds_purelight, out_crop));
      % psf_out_purelight = CropCenter(psf_ds_purelight, out_crop);
      psf_gamma = abs(psf_out .^ (1/2.2));
      psf_gamma = min(psf_gamma, 2^16 - 1);
      psf_u16 = uint16(psf_gamma);
      psf_gamma_purelight = abs(psf_out_purelight .^ (1/2.2));
      psf_gamma_purelight = min(psf_gamma_purelight, 2^16 - 1);
      psf_u16_purelight = uint16(psf_gamma_purelight);

      output_file_name = sprintf('scatterflare_aperture%04d_blur%02d_crop%02d.png', ...
                                 tt - 1, ii - 1, kk - 1);
      output_file_name_purelight = sprintf('purelight_aperture%04d_blur%02d_crop%02d.png', ...
                                 tt - 1, ii - 1, kk - 1);
      imwrite(psf_u16, strcat(out_dir, output_file_name));
      imwrite(psf_u16_purelight, strcat(out_dir_purelight, output_file_name_purelight));
      fprintf('Written to disk: %s\nAnd %s', output_file_name, output_file_name_purelight);

    end
  end
end
