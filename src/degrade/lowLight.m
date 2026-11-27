function info = lowLight(videoPath, opts)
%LOWLIGHT C1 low-light simulation (README 1.4 / 5): im2double -> x g -> noise -> uint8.
%   info = lowLight(videoPath)                         50 lux, Poisson photon noise
%   info = lowLight(v, Lux=10, Noise="combined", GaussianVar=5e-4, MaxFrames=300)
%   g = Lux / BaselineLux (baseline 300 lux unless measured). Noise:
%     "none"      attenuation + 8-bit quantisation only
%     "poisson"   photon shot noise J = poissrnd(I_att*Nphot)/Nphot, Nphot = photo-electrons
%                 at full scale (default 5000). imnoise(...,'poisson') is NOT used on doubles:
%                 it scales by 1e12 and adds almost no noise (docs/API_NOTES.md probe).
%     "gaussian"  imnoise(I_att,'gaussian',0,GaussianVar) (read noise, var 1e-4...1e-3)
%     "combined"  Poisson then Gaussian
%   Then im2uint8. Output: <OutRoot>/LowLight_C1_<lux>lux_<noise>/<subject>.mj2 (lossless) + .json.
%   info: sidecar struct (params, source, measured output incl. mean intensity ratio).
arguments
    videoPath (1,1) string {mustBeFile}
    opts.Lux (1,1) double {mustBePositive} = 50
    opts.BaselineLux (1,1) double {mustBePositive} = 300
    opts.Noise (1,1) string {mustBeMember(opts.Noise, ["none" "poisson" "gaussian" "combined"])} = "poisson"
    opts.Nphot (1,1) double {mustBePositive} = 5000
    opts.GaussianVar (1,1) double {mustBeNonnegative} = 1e-4
    opts.Seed (1,1) double = 0
    opts.Condition (1,1) string = ""
    opts.OutRoot (1,1) string = ""
    opts.Subject (1,1) string = ""
    opts.MaxFrames (1,1) double = Inf
    opts.Profile (1,1) string = "Archival"
end
if opts.OutRoot == "", opts.OutRoot = degradeDefaultRoot(); end
g = opts.Lux / opts.BaselineLux;
cond = opts.Condition;
if cond == "", cond = sprintf("LowLight_C1_%glux_%s", opts.Lux, opts.Noise); end
rng(opts.Seed, 'twister');
sumIn = 0; sumOut = 0;
    function J = apply(I, ~, ~)
        x = im2double(I) * g;
        if any(opts.Noise == ["poisson" "combined"]), x = poissrnd(x * opts.Nphot) / opts.Nphot; end
        if any(opts.Noise == ["gaussian" "combined"]), x = imnoise(x, 'gaussian', 0, opts.GaussianVar); end
        J = im2uint8(min(max(x, 0), 1));
        sumIn = sumIn + mean(double(I), 'all'); sumOut = sumOut + mean(double(J), 'all');
    end
p = struct('generator', 'lowLight', 'constraint', 'C1', 'lux', opts.Lux, ...
    'baseline_lux', opts.BaselineLux, 'gain', g, 'noise', char(opts.Noise), ...
    'nphot_full_scale', opts.Nphot, 'gaussian_var', opts.GaussianVar, 'seed', opts.Seed, ...
    'quantisation', 'im2uint8', ...
    'noise_model', 'poissrnd(I_att*Nphot)/Nphot (photon shot noise), imnoise gaussian (read noise)');
info = degradeRun(videoPath, @apply, cond, p, opts);
info.measured_mean_ratio = sumOut / max(sumIn, eps);
degradeAppend(info, 'measured_mean_ratio', info.measured_mean_ratio);
end
