function p = face_params(configPath)
%FACE_PARAMS  Parameters of the face / ROI module.
%   p = face_params()            defaults overridden by config/face_config.json
%   p = face_params(configPath)  defaults overridden by another JSON file
%   p = face_params("")          built-in defaults only
%
%   The JSON may contain any subset of the fields below. ROI boxes are
%   [u1 v1 u2 v2] fractions of the face box (u: left->right, v: top->bottom).

if nargin < 1
    configPath = fullfile(repo_root(), "config", "face_config.json");
end

p.version = "1.2";

p.det.model       = "small-network";   % RetinaFace: "small-network" | "large-network"
p.det.device      = "cpu";             % "cpu" (bit-reproducible) | "gpu" | "auto"
p.det.threshold   = 0.5;
p.det.minSize     = [10 10];           % [h w] px
p.det.every       = 10;                % full-frame detection period (frames)
p.det.everyLost   = 3;                 % detection period while the target is lost
p.det.searchScale = 2.0;               % search window size after a KLT failure (x face box)
p.det.upscale     = 1;                 % >1 upscales the frame for detection only

p.trk.targetPolicy  = "largest-central";
p.trk.centralWeight = 0.5;
p.trk.targetBox     = [];              % optional manual target box in the first frame
p.trk.minIoU        = 0.10;
p.trk.costNonAssign = 0.50;
p.trk.maxMisses     = 3;               % detection rounds before a track is deleted
p.trk.reacqDist     = 1.0;             % re-acquisition radius (face widths)
p.trk.reacqAnyAfter = 5;               % s lost before a new target may be chosen
p.trk.kfInitErr     = [200 50];
p.trk.kfMotionNoise = [100 25];
p.trk.kfMeasNoise   = 100;

p.klt.maxPts      = 300;
p.klt.minPts      = 8;
p.klt.minQuality  = 0.01;
p.klt.featInset   = [0.15 0.05];       % [x y] box fraction ignored at each side
p.klt.maxBidirErr = 2;                 % px
p.klt.maxDist     = 3;                 % px, RANSAC inlier distance
p.klt.refreshFrac = 0.5;
p.klt.reanchorIoU = 0.75;

p.roi.names    = ["forehead" "lcheek" "rcheek"];   % regions in use (order = output columns)
p.roi.forehead = [0.32 0.12 0.68 0.27];
p.roi.lcheek   = [0.15 0.50 0.38 0.72];            % viewer's left
p.roi.rcheek   = [0.62 0.50 0.85 0.72];
p.roi.glabella = [0.42 0.28 0.58 0.40];            % optional (Li et al. 2026), add to names to use
p.roi.nose     = [0.44 0.40 0.56 0.58];
p.roi.lmalar   = [0.15 0.46 0.38 0.60];
p.roi.rmalar   = [0.62 0.46 0.85 0.60];

p.skin.cb     = [77 135];              % YCbCr skin box (uint8 scale), widened for dark skin (AiPEX)
p.skin.cr     = [128 173];
p.skin.yMin   = 20;
p.skin.satMax = 250;

p.occl.reference = "forehead";         % region the relative rule compares against
p.occl.onBelow   = 0.40;               % skin ratio hysteresis (absolute rule)
p.occl.offAbove  = 0.50;
p.occl.relOn     = 0.60;               % ratio / reference ratio hysteresis (relative rule)
p.occl.relOff    = 0.70;
p.occl.emaAlpha  = 0.15;
p.occl.minPix    = 10;

p.smooth.enable    = true;
p.smooth.minCutoff = 1.0;              % Hz
p.smooth.beta      = 0.02;
p.smooth.dCutoff   = 1.0;              % Hz

p.clip.scanEvery      = 2;
p.clip.bystanderScale = 0.75;
p.clip.bystanderCrop  = 1.6;
p.clip.maskColor      = [150 190 225];
p.clip.maskPoly = [ 0.00 0.55; 0.25 0.50; 0.50 0.46; 0.75 0.50; 1.00 0.55; ...
                    1.02 0.80; 0.80 1.02; 0.50 1.10; 0.20 1.02; -0.02 0.80];

if strlength(configPath) > 0 && isfile(configPath)
    p = merge_struct(p, jsondecode(fileread(configPath)));
end
p = validate(p);
end

function a = merge_struct(a, b)
for f = string(fieldnames(b)).'
    v = b.(f);
    if isstruct(v) && isfield(a, f) && isstruct(a.(f))
        a.(f) = merge_struct(a.(f), v);
    else
        if iscell(v)
            v = string(v).';
        elseif ischar(v)
            v = string(v);
        elseif (isnumeric(v) || islogical(v)) && iscolumn(v) && ~isscalar(v)
            v = v.';
        end
        a.(f) = v;
    end
end
end

function p = validate(p)
p.roi.names = string(p.roi.names);
for n = p.roi.names
    if ~isfield(p.roi, n) || numel(p.roi.(n)) ~= 4
        error("face_params:roi", "Region ""%s"" needs a 1x4 [u1 v1 u2 v2] entry in roi.", n);
    end
    p.roi.(n) = double(p.roi.(n)(:).');
end
if ~any(p.roi.names == p.occl.reference)
    warning("face_params:reference", "occl.reference ""%s"" is not a region; relative occlusion rule disabled.", p.occl.reference);
end
if isempty(p.trk.targetBox), p.trk.targetBox = []; end
end
