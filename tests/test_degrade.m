function tests = test_degrade
%TEST_DEGRADE Constraint generators C1, C4, C5, C6, C7 on a synthetic video (no real faces).
%   Checks: output resolution / fps / frame count, sidecar JSON fields, C1 mean intensity
%   scales by g = lux/300, C5 measured fps and logged bitrate, C7 mask covers >= 40 % of the
%   lower-face box, C6 changes only the forehead / cheek regions.
tests = functiontests(localfunctions);
end

function setupOnce(tc)
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'config')); addpath(genpath(fullfile(root, 'src')));
cfg = getConfig();
base = fullfile(fileparts(cfg.datasetRoot), 'work', 'degraded');   % generated videos live here only
if ~isfolder(base), base = tempdir; end
tc.TestData.out = fullfile(base, ['_unittest_' char(java.util.UUID.randomUUID)]);
mkdir(tc.TestData.out);
% synthetic 640x480, 30 fps, 60 frames: gradient background + skin-coloured ellipse "face"
% with a 1.2 Hz brightness modulation and mild noise
H = 480; W = 640; n = 60; fs = 30;
[x, y] = meshgrid(1:W, 1:H);
face = ((x - 320) / 110).^2 + ((y - 240) / 150).^2 <= 1;
tc.TestData.box = [210 90 220 300];
src = fullfile(tc.TestData.out, 'synthetic.avi');
w = VideoWriter(src, 'Uncompressed AVI'); w.FrameRate = fs; open(w);
rng(7);
for k = 1:n
    I = repmat(0.25 + 0.4 * x / W, 1, 1, 3);
    skin = reshape([0.78 0.57 0.45] * (1 + 0.01 * sin(2 * pi * 1.2 * (k - 1) / fs)), 1, 1, 3);
    I = I .* ~face + skin .* face + 0.01 * randn(H, W, 3);
    writeVideo(w, im2uint8(min(max(I, 0), 1)));
end
close(w);
tc.TestData.src = src; tc.TestData.n = n; tc.TestData.fs = fs;
end

function teardownOnce(tc)
if isfolder(tc.TestData.out), rmdir(tc.TestData.out, 's'); end
end

function checkVideo(tc, info, H, W, fps, n)
v = VideoReader(info.output.path);
tc.verifyEqual([v.Height v.Width], [H W]);
tc.verifyEqual(v.FrameRate, fps, 'AbsTol', 1e-3);
tc.verifyEqual(v.NumFrames, n);
tc.verifyTrue(isfile(info.sidecar));
s = jsondecode(fileread(info.sidecar));
tc.verifyTrue(all(isfield(s.output, {'width', 'height', 'fps', 'duration_s', 'bitrate_kbps'})));
tc.verifyEqual([s.output.height s.output.width], [H W]);
tc.verifyTrue(startsWith(info.output.path, tc.TestData.out));
end

function testLowLightMeanScalesByGain(tc)
for lux = [50 10]
    info = lowLight(tc.TestData.src, Lux=lux, Noise="none", OutRoot=tc.TestData.out);
    checkVideo(tc, info, 480, 640, 30, tc.TestData.n);
    g = lux / 300;
    tc.verifyEqual(info.measured_mean_ratio, g, 'RelTol', 0.03);
    % also from the written (lossless) file
    v = VideoReader(info.output.path); a = VideoReader(tc.TestData.src);
    r = mean(double(readFrame(v)), 'all') / mean(double(readFrame(a)), 'all');
    tc.verifyEqual(r, g, 'RelTol', 0.03);
end
info = lowLight(tc.TestData.src, Lux=25, Noise="combined", GaussianVar=1e-4, OutRoot=tc.TestData.out);
tc.verifyEqual(info.measured_mean_ratio, 25 / 300, 'RelTol', 0.05);   % zero-mean noise keeps the mean
s = jsondecode(fileread(info.sidecar));
tc.verifyEqual(s.params.gain, 25 / 300, 'AbsTol', 1e-12);
tc.verifyEqual(s.params.noise, 'combined');
end

function testLowResSizes(tc)
info = lowRes(tc.TestData.src, Size=[240 320], OutRoot=tc.TestData.out);
checkVideo(tc, info, 240, 320, 30, tc.TestData.n);
info = lowRes(tc.TestData.src, Size=[120 160], OutRoot=tc.TestData.out);
checkVideo(tc, info, 120, 160, 30, tc.TestData.n);
end

function testLowBandwidthFpsAndBitrate(tc)
for fps = [10 6]
    info = lowBandwidth(tc.TestData.src, Fps=fps, BitrateKbps=500, OutRoot=tc.TestData.out);
    nExp = floor((tc.TestData.n - 1) / tc.TestData.fs * fps) + 1;
    checkVideo(tc, info, 480, 640, fps, nExp);
    tc.verifyEqual(info.fps_measured, fps, 'RelTol', 0.01);
    tc.verifyGreaterThan(info.output.bitrate_kbps, 0);
    s = jsondecode(fileread(info.sidecar));
    tc.verifyTrue(isfield(s.params, 'quality_attempts'));
    tc.verifyEqual(numel(s.params.source_t_s), nExp);
    tc.verifyEqual(diff(s.params.source_t_s(:)), repmat(1 / fps, nExp - 1, 1), 'AbsTol', 0.5 / tc.TestData.fs);
    if info.target_met, tc.verifyLessThanOrEqual(info.output.bitrate_kbps, 500); end
end
end

function testMaskCoversLowerFace(tc)
for variant = ["surgical" "cloth"]
    info = mask(tc.TestData.src, Variant=variant, FaceBox=tc.TestData.box, OutRoot=tc.TestData.out);
    checkVideo(tc, info, 480, 640, 30, tc.TestData.n);
    tc.verifyGreaterThanOrEqual(info.coverage_min, 0.40);
    % independent pixel check: share of the lower-face box (rows 0.60-1.0 of the box) painted
    v = VideoReader(info.output.path); J = double(readFrame(v));
    b = tc.TestData.box;
    ys = round(b(2) + 0.60 * b(4)):round(b(2) + b(4)); xs = round(b(1)):round(b(1) + b(3));
    s = jsondecode(fileread(info.sidecar)); col = reshape(s.params.color_rgb, 1, 1, 3);
    painted = all(abs(J(ys, xs, :) - col) <= 2, 3);
    tc.verifyGreaterThanOrEqual(mean(painted(:)), 0.40);
end
end

function testAgingIsLocal(tc)
info = aging(tc.TestData.src, Strength=0.2, FaceBox=tc.TestData.box, OutRoot=tc.TestData.out);
checkVideo(tc, info, 480, 640, 30, tc.TestData.n);
v = VideoReader(info.output.path); a = VideoReader(tc.TestData.src);
J = double(readFrame(v)); I = double(readFrame(a));
b = tc.TestData.box;
fh = I(round(b(2) + 0.1 * b(4)):round(b(2) + 0.25 * b(4)), round(b(1) + 0.3 * b(3)):round(b(1) + 0.7 * b(3)), :);
fj = J(round(b(2) + 0.1 * b(4)):round(b(2) + 0.25 * b(4)), round(b(1) + 0.3 * b(3)):round(b(1) + 0.7 * b(3)), :);
tc.verifyLessThan(mean(fj(:)), mean(fh(:)));                       % forehead darkened by texture
tc.verifyLessThanOrEqual(max(fj(:) - fh(:)), 1);                     % shadows only
tc.verifyLessThanOrEqual(1 - mean(fj(:)) / mean(fh(:)), 0.2);        % bounded by the strength
out = J(1:round(b(2)) - 10, :, :); in = I(1:round(b(2)) - 10, :, :);
tc.verifyEqual(out, in);                                             % untouched outside the face
end
