%  // clang-format off
function psf = GetPsf(aperture, phase, wavelengths, spectral_response, crop)
% GetPsf Computes the RGB response of an aperture under a point light source.
%
% psf = GetPsf(aperture, phase, wavelengths, spectral_response, crop)
% Computes the point spread function (PSF) of the given aperture in response to
% a white point light source (i.e., it has a flat spectrum).
%
% Arguments
%
% aperture: A grayscale image representing the aperture, where 0 means total
%           opacity and 1 means total transparency.
%
% phase: An image of the same size as `aperture` representing the phase shift.
%
% wavelengths: An L-vector of wavelengths at which the light spectrum is
%              sampled. They are normalized by the wavelength at which `phase`
%              is computed.
%
% spectral_response: Sensitivity of RGB pixels at `wavelengths`. Size [3, L].
%
% crop: Side length of the output array. It should be smaller than the input due
%       to wavelength-dependent resizing - otherwise we would get out-of-range
%       samples.
%
% Returns
%
% psf: An RGB image of size [crop, crop].
%
% Required toolboxes: none.
%
% Implementation notes
% - Accounts for wavelength-dependent phase by expanding `phase` along the
%   spectrum dimension.
% - Computes a per-wavelength PSF via FFT of the complex pupil function.
% - Applies wavelength-dependent spatial scaling, crops to `crop`, then mixes
%   wavelengths into RGB using `spectral_response`.




% Expand to 3-D array of size [H, W, C] where C is the size of the `wavelengths`
% vector (i.e., the number of samples in the spectrum). This accounts for the
% phase term's dependency on the wavelength.
phase_wl = phase ./ reshape(wavelengths, 1, 1, []);

% Pupil function in the frequency domain.
pupil_wl = aperture .* exp(1j * phase_wl);

% Point spread function (PSF) in the spatial domain is related to the pupil
% function in the frequency domain.
num_wl = length(wavelengths);
