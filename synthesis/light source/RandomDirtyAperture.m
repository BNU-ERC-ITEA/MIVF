%  // clang-format off
function im = RandomDirtyAperture(mask, source_t, streak_angle)
% RandomDirtyAperture Synthetic dirty aperture with random dots and scratches.
%
% im = RandomDirtyAperture(mask, source_t, streak_angle)
% Returns an N x N monochromatic image emulating a dirty aperture plane.
% Specifically, we add disks and polylines of random size and opacity to an
% otherwise white image, in an attempt to model random dust and scratches. 
%
% TODO(qiurui): the spatial scale of the random dots and polylines are currently
%   hard-coded in order to match the paper. They should instead be relative to
%   the requested resolution, n.
%
% Arguments
%
% mask: An [N, N]-logical matrix representing the aperture mask. Typically, this
%       should be a centered disk of 1 surrounded by 0.
% source_t: Type of contamination ('none', 'dirty', 'streak')
% streak_angle: Angle for streak contamination (in radians), only used when source_t='streak'
%
% Returns
%
% im: An [N, N]-matrix of values in [0, 1] where 0 means completely opaque and 1
%     means completely transparent. The returned matrix is real-valued (i.e., we
%     ignore the phase shift that may be introduced by the "dust" and
%     "scratches").
%
% Required toolboxes: Computer Vision Toolbox.
%
% Notes
% - Creates an all-white image and injects random disks (dust) and polylines
%   (scratches) with random size/opacity.
% - Multiplies the result by `mask` so contamination only appears inside the
%   transmissive aperture region.
% - Helper `RandomPointsInUnitCircle` generates random 2D steps for polylines.

n = size(mask, 1);
im = ones(size(mask), 'single');

% Set default streak_angle if not provided
if nargin < 3
    streak_angle = 0;
end

if strcmp(source_t, 'none')
    im = single(mask) .* im;
elseif strcmp(source_t, 'dirty')
    %% Add dots (circles), simulating dust.
    num_dots = max(0, round(50 + randn * 20));
    max_radius = max(0, 75 + randn * 25);
    for i = 1:num_dots
      circle_xyr = rand(1, 3, 'single') .* [n, n, max_radius];
      opacity = 0.5 + rand * 0.5;
      im = insertShape(im, 'FilledCircle', circle_xyr, 'Color', 'black', ...
                      'Opacity', opacity);
    end
    
    %% Add polylines, simulating scratches.

    num_lines = max(0, round(30 + randn * 5));
    max_width = max(0, round(15 + randn * 5));
    for i = 1:num_lines
      num_segments = randi(16);
      start_xy = rand(2, 1) * n;
      segment_length = rand * 200;
      segments_xy = RandomPointsInUnitCircle(num_segments) * segment_length;
      vertices_xy = cumsum([start_xy, segments_xy], 2);
      vertices_xy = reshape(vertices_xy, 1, []);
      width = randi(max_width);
      % Note: the 'Opacity' option doesn't apply to lines, so we have to change the
      % line color to achieve a similar effect. Also note that [0.5 .. 1] opacity
      % maps to [0.5 .. 0] in color values.
      color = rand * 0.5;
      im = insertShape(im, 'Line', vertices_xy, 'LineWidth', width, ...
                       'Color', [color, color, color]);
    end
    

    % %% Add dominant streak scratches (optional)
    % 
    % % Get image dimensions
    % height = n;
    % width_1 = n;
    % 
    % % Define scratch parameters
    % scratch_min_length = max(150, round(width_1 / 2)); % Ensure minimum length is at least half the width
    % scratch_max_length = 2 * scratch_min_length;
    % scratch_width = randi(10);
    % 
    % % Determine the basic direction of scratches
    % basic_angle = pi*rand;
    % 
    % % Loop to add scratches
    % num_scratches = 150; % Number of scratches to add
    % for i = 1:num_scratches
    %     % Randomly select scratch start point
    %     x_start = randi(width_1);
    %     y_start = randi(height);
    % 
    %     % Randomize scratch length and direction
    %     scratch_length = scratch_min_length + randi(scratch_max_length - scratch_min_length);
    %     direction_angle = basic_angle + randn * 0.02; % Reduced angle deviation from horizontal
    % 
    %     % Calculate end point of scratch
    %     dx = scratch_length * cos(direction_angle);
    %     dy = scratch_length * sin(direction_angle);
    % 
    %     x_end = x_start + dx;
    %     y_end = y_start + dy;
    % 
    %     % Check if the end point is out of bounds
    %     if x_end > width_1 || x_end < 1 || y_end > height || y_end < 1
    %         % Adjust length to keep both ends inside the image
    %         while x_end > width_1 || x_end < 1 || y_end > height || y_end < 1
    %             scratch_length = scratch_length * 0.9; 
    %             dx = scratch_length * cos(direction_angle);
    %             dy = scratch_length * sin(direction_angle);
    %             x_end = x_start + dx;
    %             y_end = y_start + dy;
    %         end
    %     end
    % 
    %     % Create scratch coordinates
    %     vertices_xy = [x_start, y_start, x_end, y_end];
    %     % fprintf('index:%d  , vertice_xy:\n',i)
    %     % disp(vertices_xy)
    % 
    %     % Apply scratch to image
    %     width_s = scratch_width + randi(5);
    %     im = insertShape(im, 'Line', vertices_xy, 'LineWidth', width_s, ...
    %                      'Color', [0, 0, 0]); % Black color for scratches
    % end
    % 
    % im = imrotate(im, 180); % Rotate and process again to keep symmetry
    % 
    % for i = 1:num_scratches
    %     % Randomly select scratch start point
    %     x_start = randi(width_1);
    %     y_start = randi(height);
    % 
    %     % Randomize scratch length and direction
    %     scratch_length = scratch_min_length + randi(scratch_max_length - scratch_min_length);
    %     direction_angle = basic_angle + randn * 0.02; % Reduced angle deviation from horizontal
    % 
    %     % Calculate end point of scratch
    %     dx = scratch_length * cos(direction_angle);
    %     dy = scratch_length * sin(direction_angle);
    % 
    %     x_end = x_start + dx;
    %     y_end = y_start + dy;
    % 
    %     % Check if the end point is out of bounds
    %     if x_end > width_1 || x_end < 1 || y_end > height || y_end < 1
    %         % Adjust length to keep both ends inside the image
    %         while x_end > width_1 || x_end < 1 || y_end > height || y_end < 1
    %             scratch_length = scratch_length * 0.9; 
    %             dx = scratch_length * cos(direction_angle);
    %             dy = scratch_length * sin(direction_angle);
    %             x_end = x_start + dx;
    %             y_end = y_start + dy;
    %         end
    %     end
    % 
    %     % Create scratch coordinates
    %     vertices_xy = [x_start, y_start, x_end, y_end];
    %     % fprintf('index:%d  , vertice_xy:\n',i)
    %     % disp(vertices_xy)
    % 
    %     % Apply scratch to image
    %     width_s = scratch_width + randi(5);
    %     im = insertShape(im, 'Line', vertices_xy, 'LineWidth', width_s, ...
    %                      'Color', [0, 0, 0]); % Black color for scratches
    % end

    im = single(mask) .* rgb2gray(im);

elseif strcmp(source_t, 'streak')
    %% Add streak contamination
    % Get image dimensions
    height = n;
    width_1 = n;
    
    % Define scratch parameters
    scratch_min_length = max(150, round(width_1 / 2));
    scratch_max_length = 2 * scratch_min_length;
    scratch_width = randi(10);
    
    % Use provided angle for streaks
    basic_angle = streak_angle;
    
    % Loop to add scratches
    num_scratches = 150;
    for i = 1:num_scratches
        % Randomly select scratch start point
        x_start = randi(width_1);
        y_start = randi(height);

        % Randomize scratch length
        scratch_length = scratch_min_length + randi(scratch_max_length - scratch_min_length);
        direction_angle = basic_angle + randn * 0.02;

        % Calculate end point of scratch
        dx = scratch_length * cos(direction_angle);
        dy = scratch_length * sin(direction_angle);

        x_end = x_start + dx;
        y_end = y_start + dy;

        % Check if the end point is out of bounds
        if x_end > width_1 || x_end < 1 || y_end > height || y_end < 1
            % Adjust length to keep both ends inside the image
            while x_end > width_1 || x_end < 1 || y_end > height || y_end < 1
                scratch_length = scratch_length * 0.9; 
                dx = scratch_length * cos(direction_angle);
                dy = scratch_length * sin(direction_angle);
                x_end = x_start + dx;
                y_end = y_start + dy;
            end
        end

        % Create scratch coordinates
        vertices_xy = [x_start, y_start, x_end, y_end];

        % Apply scratch to image
        width_s = scratch_width + randi(5);
        im = insertShape(im, 'Line', vertices_xy, 'LineWidth', width_s, ...
                         'Color', [0, 0, 0]);
    end
    
    im = imrotate(im, 180); % Rotate and process again to keep symmetry
    
    for i = 1:num_scratches
        x_start = randi(width_1);
        y_start = randi(height);

        scratch_length = scratch_min_length + randi(scratch_max_length - scratch_min_length);
        direction_angle = basic_angle + randn * 0.02;

        dx = scratch_length * cos(direction_angle);
        dy = scratch_length * sin(direction_angle);

        x_end = x_start + dx;
        y_end = y_start + dy;

        if x_end > width_1 || x_end < 1 || y_end > height || y_end < 1
            while x_end > width_1 || x_end < 1 || y_end > height || y_end < 1
                scratch_length = scratch_length * 0.9; 
                dx = scratch_length * cos(direction_angle);
                dy = scratch_length * sin(direction_angle);
                x_end = x_start + dx;
                y_end = y_start + dy;
            end
        end

        vertices_xy = [x_start, y_start, x_end, y_end];

        width_s = scratch_width + randi(5);
        im = insertShape(im, 'Line', vertices_xy, 'LineWidth', width_s, ...
                         'Color', [0, 0, 0]);
    end

    im = single(mask) .* rgb2gray(im);

end


end

function xy = RandomPointsInUnitCircle(num_points)
r = rand(1, num_points, 'single');
theta = rand(1, num_points, 'single') * 2 * pi;
xy = [r .* cos(theta); r .* sin(theta)];
end
