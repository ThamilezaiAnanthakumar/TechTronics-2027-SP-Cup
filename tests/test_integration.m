function test_integration()
%TEST_INTEGRATION Face-module output -> Kalman -> signal chain, on synthetic data.
%   No video or face detector needed. Builds a process_video-shaped struct
%   with a 72 bpm pulse, jittered frame times, a 4 s lost-target gap and an
%   occluded right cheek, then runs subjectHR and checks:
%     - both denoise variants recover 72 bpm on 'face' (MAE < 1 bpm)
%     - windows overlapping the gap are rejected (NaN), others kept
%     - the occluded rcheek has no valid windows

repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repoRoot, 'config'));
addpath(genpath(fullfile(repoRoot, 'src')));
cfg = getConfig();
rng(1);

fs = 30; dur = 60; hrTrue = 72;
n = dur * fs;
t = (0:n-1)' / fs + 0.002 * randn(n, 1);  t(1) = 0;  t = sort(t);
pulse = sin(2*pi*hrTrue/60*t);
base = [150 110 90];
names = ["forehead" "lcheek" "rcheek"];
fv.t = t;
fv.roi_names = names;
fv.npix = repmat([3000 2000 2000], n, 1);
fv.occluded = false(n, 3);
fv.quality = 2 * ones(n, 1);
for r = 1:3
    % noise ~ pixel noise (~5 levels) / sqrt(~2000 skin pixels) in a ROI mean
    rgb = base .* (1 + 0.004 * pulse .* [0.3 1 0.5]) + 0.15 * randn(n, 3);
    fv.("rgb_" + names(r)) = rgb;
end
fv.occluded(:, 3) = true;                       % right cheek covered
lost = t >= 30 & t < 34;                        % target lost for 4 s
fv.quality(lost) = 0;
num = zeros(n, 3); den = zeros(n, 1);
for r = 1:2
    num = num + fv.npix(:, r) .* fv.("rgb_" + names(r));
    den = den + fv.npix(:, r);
end
fv.rgb_face = num ./ den;
for r = 1:3, fv.("rgb_" + names(r))(lost, :) = NaN; end
fv.rgb_face(lost, :) = NaN;

% contact PPG at 60 Hz with the same pulse
tg = (0:1/60:dur)';
gt = struct('t', tg, 'ppg', sin(2*pi*hrTrue/60*tg) + 0.05*randn(size(tg)), ...
            'hrSensor', hrTrue * ones(size(tg)));

[T, R] = subjectHR(fv, gt, cfg);
disp(T(T.Method == "POS", :));

for d = string(cfg.denoiseSet)
    row = T(T.Denoise == d & T.Method == "POS" & T.ROI == "face", :);
    assert(row.MAE < 1, 'POS/face/%s MAE %.2f >= 1 bpm', d, row.MAE);
    assert(row.coverage < 100 && row.coverage > 60, ...
        'coverage %.1f%%: gap windows should be dropped, the rest kept', row.coverage);
    rc = T(T.Denoise == d & T.Method == "POS" & T.ROI == "rcheek", :);
    assert(rc.nWin == 0, 'occluded rcheek should have no valid windows');
end
i = find(strcmp({R.denoise}, 'kalman') & strcmp({R.method}, 'POS') & strcmp({R.roi}, 'face'), 1);
gapWin = abs(R(i).tHR - 32) < 5;               % windows fully overlapping the gap
assert(all(isnan(R(i).hr(gapWin))), 'windows over the lost gap must be NaN');
fprintf('test_integration: PASS\n');
end
