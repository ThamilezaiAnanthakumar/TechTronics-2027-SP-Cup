function T = round1_smoke_degrade(opts)
%ROUND1_SMOKE_DEGRADE Smoke run of the C1/C4/C5/C6/C7 generators on ONE UBFC dev subject.
%   T = round1_smoke_degrade()                         subject1, first 300 frames
%   Videos + sidecars: <work>\degraded\<condition>\subject1.*  (never inside the repo).
%   Summary: outputs/round1_smoke/degrade_summary.csv (git-ignored).
arguments
    opts.Subject (1,1) string = "subject1"
    opts.MaxFrames (1,1) double = 300
end
dev = "subject" + [1 3 5 8:16];
assert(ismember(opts.Subject, dev), 'Only UBFC dev subjects {1,3,5,8-16} may be used.');
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'config')); addpath(genpath(fullfile(root, 'src')));
cfg = getConfig(); cfg.subject = char(opts.Subject);
vp = loadUBFC(cfg);
n = opts.MaxFrames;
runs = {
    @() lowLight(vp, Lux=50, Noise="poisson", MaxFrames=n)
    @() lowLight(vp, Lux=25, Noise="poisson", MaxFrames=n)
    @() lowLight(vp, Lux=10, Noise="poisson", MaxFrames=n)
    @() lowLight(vp, Lux=50, Noise="gaussian", GaussianVar=1e-3, MaxFrames=n)
    @() lowLight(vp, Lux=10, Noise="combined", GaussianVar=1e-4, MaxFrames=n)
    @() lowRes(vp, Size=[240 320], MaxFrames=n)
    @() lowRes(vp, Size=[120 160], MaxFrames=n)
    @() lowBandwidth(vp, Fps=10, BitrateKbps=500, MaxFrames=n)
    @() lowBandwidth(vp, Fps=8, BitrateKbps=500, MaxFrames=n)
    @() lowBandwidth(vp, Fps=6, BitrateKbps=500, MaxFrames=n)
    @() lowBandwidth(vp, Fps=10, BitrateKbps=250, MaxFrames=n)
    @() aging(vp, Strength=0.1, MaxFrames=n)
    @() aging(vp, Strength=0.2, MaxFrames=n)
    @() mask(vp, Variant="surgical", MaxFrames=n)
    @() mask(vp, Variant="cloth", MaxFrames=n)
    };
rows = cell(numel(runs), 9);
for i = 1:numel(runs)
    tic; info = runs{i}(); dt = toc;
    o = info.output; extra = "";
    if isfield(info, 'measured_mean_ratio'), extra = sprintf("mean ratio %.4f (g %.4f)", info.measured_mean_ratio, info.params.gain); end
    if isfield(info, 'fps_measured'), extra = sprintf("fps meas %.3f, quality %g, target met %d", info.fps_measured, info.params.quality, info.target_met); end
    if isfield(info, 'coverage_mean'), extra = sprintf("lower-face coverage mean %.2f min %.2f, no-face frames %d", info.coverage_mean, info.coverage_min, info.frames_without_face); end
    if isfield(info, 'mean_darkening_in_region'), extra = sprintf("mean darkening %.3f, no-face frames %d", info.mean_darkening_in_region, info.frames_without_face); end
    rows(i, :) = {string(info.condition), o.width, o.height, o.fps, o.num_frames, o.duration_s, o.bitrate_kbps, extra, dt};
    fprintf('%-32s %4dx%-4d %5.2f fps %4d fr %6.2f s %9.1f kbps  %s  (%.1f s)\n', rows{i, :});
end
T = cell2table(rows, 'VariableNames', {'Condition', 'Width', 'Height', 'Fps', 'Frames', 'Duration_s', 'Bitrate_kbps', 'Measured', 'Runtime_s'});
outDir = fullfile(root, 'outputs', 'round1_smoke'); if ~isfolder(outDir), mkdir(outDir); end
writetable(T, fullfile(outDir, 'degrade_summary.csv'));
end
