function test_synthetic()
%TEST_SYNTHETIC Sanity check (README 6.1): known 72 bpm pulse in synthetic RGB.
%   60 s at 30 fps, 1.2 Hz pulse (0.5% of mean on G, smaller on R and B),
%   multiplicative illumination drift and Gaussian noise. Every method must
%   give 72 +/- 1 bpm in every window. Errors if not.

repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repoRoot, 'config'));
addpath(genpath(fullfile(repoRoot, 'src')));

rng(0);
cfg = getConfig();
fs = 30; T = 60; f0 = 1.2;
t = (0:1/fs:T-1/fs)';
N = numel(t);

base  = [150 110 80];                       % mean R G B
amp   = [0.002 0.005 0.001];                % relative pulse amplitude per channel
pulse = sin(2*pi*f0*t);
drift = 1 + 0.03*sin(2*pi*0.05*t) + 0.02*t/T;   % slow illumination change
% Noise on the ROI *spatial mean*: ~2 grey levels per pixel averaged over
% ~2000 pixels -> ~0.05. (At 0.2, CHROM/POS still give median 72.0 bpm but
% single 10 s windows wander by ~1.1 bpm; noise-free error is < 0.01 bpm.)
noise = 0.05 * randn(N, 3);

rgb = (base .* (1 + pulse .* amp)) .* drift + noise;

tol = 1;
fprintf('Synthetic test: expected %.1f bpm\n', 60*f0);
allOK = true;
for k = 1:numel(cfg.methods)
    method = cfg.methods{k};
    p  = rgbToPulse(rgb, fs, method, cfg);
    pf = preprocessPulse(p, fs, cfg);
    [~, hr] = estimateHR(pf, fs, cfg);
    err = max(abs(hr - 60*f0));
    ok = err <= tol;
    allOK = allOK && ok;
    fprintf('  %-6s median %.2f bpm, max |err| %.2f bpm over %d windows  [%s]\n', ...
        method, median(hr), err, numel(hr), passFail(ok));
end
if ~allOK
    error('test_synthetic:fail', 'Synthetic test FAILED.');
end
fprintf('Synthetic test PASSED.\n');
end

function s = passFail(ok)
if ok, s = 'PASS'; else, s = 'FAIL'; end
end
