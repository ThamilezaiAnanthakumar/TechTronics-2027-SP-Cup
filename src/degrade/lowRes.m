function info = lowRes(videoPath, opts)
%LOWRES C4 low-resolution simulation (README 1.4 / 5): imresize bicubic.
%   info = lowRes(videoPath)                  320x240
%   info = lowRes(v, Size=[120 160], MaxFrames=300)    Size = [H W] (160x120 is the stress point)
%   Output: <OutRoot>/LowRes_C4_<W>x<H>/<subject>.mj2 (lossless) + .json sidecar.
arguments
    videoPath (1,1) string {mustBeFile}
    opts.Size (1,2) double {mustBePositive, mustBeInteger} = [240 320]
    opts.Condition (1,1) string = ""
    opts.OutRoot (1,1) string = ""
    opts.Subject (1,1) string = ""
    opts.MaxFrames (1,1) double = Inf
    opts.Profile (1,1) string = "Archival"
end
if opts.OutRoot == "", opts.OutRoot = degradeDefaultRoot(); end
cond = opts.Condition;
if cond == "", cond = sprintf("LowRes_C4_%dx%d", opts.Size(2), opts.Size(1)); end
p = struct('generator', 'lowRes', 'constraint', 'C4', 'size_hw', opts.Size, ...
    'method', 'imresize bicubic (antialiasing on, MATLAB default)');
info = degradeRun(videoPath, @(I, ~, ~) imresize(I, opts.Size, 'bicubic'), cond, p, opts);
end
