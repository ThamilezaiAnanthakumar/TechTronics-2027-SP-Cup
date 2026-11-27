function test_viterbi()
%TEST_VITERBI viterbiHR follows the true HR track despite harmonic and random spikes (README 10).
repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(repoRoot, 'src')));
rng(3);
f = (0.5:0.005:3.2)'; nW = 60;
hrTrue = 75 + 10 * sin(2 * pi * (1:nW)' / 60);           % slow 65-85 bpm drift
P = zeros(nW, numel(f));
for w = 1:nW
    f0 = hrTrue(w) / 60;
    P(w, :) = exp(-0.5 * ((f' - f0) / 0.03).^2) + 0.3 * rand(1, numel(f)) .* (rand(1, numel(f)) > 0.97);
    if mod(w, 4) == 0                                    % every 4th window: a stronger 2x harmonic
        P(w, :) = P(w, :) + 1.6 * exp(-0.5 * ((f - 2 * f0) / 0.03).^2)';
    end
end
[~, k] = max(P, [], 2); argmaxHr = 60 * f(k);
[hr, conf] = viterbiHR(P, f);
errV = max(abs(hr - hrTrue)); errA = max(abs(argmaxHr - hrTrue));
fprintf('argmax max |err| %.1f bpm, Viterbi max |err| %.2f bpm, mean conf %.2f\n', errA, errV, mean(conf));
assert(errA > 40, 'test setup: argmax should be fooled by the harmonic spikes');
assert(errV < 1.0, 'Viterbi must stay on the true track (max err %.2f)', errV);
fprintf('test_viterbi: PASS\n');
end
