function test_troika_track()
%TEST_TROIKA_TRACK troikaTrack suppresses motion bursts and still follows real HR changes.
%   Synthetic fused spectra, 1 s step: true HR 55 bpm for 40 s, then a real rise to 70 bpm at
%   0.5 bpm/s, then 70 bpm. Every 7th window a motion peak at 85 bpm and every 11th at 65 bpm is
%   1.5x stronger than the pulse (the operator's observation: 55 -> 85 -> 65 jumps).
repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(repoRoot, 'src')));
rng(5);
f = (0.7:0.0037:3.0)'; nW = 120;
hrTrue = [55 * ones(40, 1); (55:0.5:70)'; 70 * ones(nW - 71, 1)];
P = zeros(nW, numel(f));
g = @(c) exp(-0.5 * ((60 * f' - c) / 1.5).^2);
for w = 1:nW
    P(w, :) = g(hrTrue(w)) + 0.4 * g(2 * hrTrue(w)) + 0.05 * rand(1, numel(f));
    if mod(w, 7) == 0, P(w, :) = P(w, :) + 1.5 * g(85); end
    if mod(w, 11) == 0, P(w, :) = P(w, :) + 1.5 * g(65); end
end
[~, k] = max(P, [], 2); argmaxHr = 60 * f(k);
[hr, info] = troikaTrack(P, f);
errA = max(abs(argmaxHr - hrTrue)); errT = max(abs(hr - hrTrue));
stepT = max(abs(diff(hr)));
fprintf('argmax: max |err| %.1f bpm, max step %.1f | TROIKA: max |err| %.2f bpm, max step %.2f bpm, actions %s\n', ...
    errA, max(abs(diff(argmaxHr))), errT, stepT, strjoin(unique(info.action)', ','));
assert(errA > 20, 'setup: argmax must be fooled by the motion peaks');
assert(stepT < 5, 'consecutive readings must never change by 5 bpm or more (max %.2f)', stepT);
assert(errT < 2.5, 'tracked HR must stay within 2.5 bpm of the truth, incl. the real rise (max %.2f)', errT);
fprintf('test_troika_track: PASS\n');
end
