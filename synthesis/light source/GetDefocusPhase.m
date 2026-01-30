%  // clang-format off
function [phase, mask] = GetDefocusPhase(n, r, aperture_t)
% GetDefocusPhase Phase shift due to defocus for a round aperture.
%
% [phase, mask] = GetDefocusPhase(n, aperture_r)
% Computes the phase shift per unit defocus in the Fourier domain. Also returns
% the corresponding circular mask on the Fourier plane that defines the valid
% region of the frequency response.
%
% Arguments
%
% n: Number of samples in each direction for the image and spectrum. The output
%    will be an [n, n]-array.
%
% r: Radius of the circular low-pass filter applied on the spectrum, assuming 
%    the spectrum is a unit square.
%
% Returns
%
% phase: Amount of (complex) phase shift in the spectrum for each unit (1) of
%        defocus. Zero outside the disk of radius `r`. [n, n]-array.
%
% mask: A centered disk of 1 surrounded by 0, representing the low-pass filter
%       that is applied to the spectrum (including the `phase` array above).
%       [n, n]-array.
%
% Required toolboxes: none.
%
% Notes
% - Builds a low-pass aperture mask (circle or polygon) on the Fourier plane.
% - Uses the Zernike polynomial (degree 2, order 0) to model defocus.
% - Sets phase outside the mask to 0.

sample_x = linspace(-(n - 1) / 2, (n - 1) / 2, n) / n / r;
[xx, yy] = meshgrid(sample_x);
[~, rr] = cart2pol(xx, yy);

if strcmp(aperture_t, 'circle')
    %% Pixel center coordinates in Cartesian and polar forms.

    %% The mask is simply a centered unit disk.
    % Zernike polynomials below are only defined on the unit disk.
    mask = rr <= 1;

    % figure;
    % imshow(mask); % Assumes `mask` is a 2D array suitable for imshow (binary/gray)
    % title('Mask Visualization');


elseif strcmp(aperture_t, 'poly_6')
    
    % Define the center of the hexagon at the origin (for simplicity)
    center = [n/2, n/2];
    
    % Radius of the circumscribed circle (which equals the distance from the center to any vertex)
    radius = n * r;
    
    % Calculate vertices of the hexagon
    angleIncrement = pi/3;
    verticesX = radius * cos([0:5]*angleIncrement) + center(1);
    verticesY = radius * sin([0:5]*angleIncrement) + center(2);
    
    % Create a grid of coordinates
    [xGrid, yGrid] = meshgrid(linspace(-n/2, n/2, n) + center(1), linspace(-n/2, n/2, n) + center(2));
    
    % Use inpolygon to determine which points are inside the hexagon
    inHexagon = inpolygon(xGrid(:), yGrid(:), verticesX, verticesY);
    mask = reshape(inHexagon, size(xGrid));

else
    error('Invalid aperture shape. Choose ''circle'' or ''poly_6''.');
end


%% Compute the Zernike polynomial of degree 2, order 0. 
% Zernike polynomials form a complete, orthogonal basis over the unit disk. The 
% "degree 2, order 0" component represents defocus, and is defined as (in 
% unnormalized form):
%
%     Z = 2 * r^2 - 1.
%
% Reference:
% Paul Fricker (2021). Analyzing LASIK Optical Data Using Zernike Functions.
% https://www.mathworks.com/company/newsletters/articles/analyzing-lasik-optical-data-using-zernike-functions.html
phase = single(2 * rr .^ 2 - 1);
phase(~mask) = 0;

end
