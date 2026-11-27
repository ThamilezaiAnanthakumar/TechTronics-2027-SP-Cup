function tests = test_round1
%TEST_ROUND1 process_round1 / runOffline end to end on synthetic patch traces (no real faces).
%   10 anatomical patches (shared/patches.json names/groups), 72 bpm pulse, 40 s at ~29.5 fps,
%   jitter-free VideoReader-like timestamps; a 4 s no-face gap; group-C patches occluded in
%   the last 15 s (mask). Checks CSV columns/types, the six .mat variables and their types,
%   status codes (OK / NO_FACE / MASKED), HR accuracy, MaxFramesInMat, Viterbi option.
tests = functiontests(localfunctions);
end

function setupOnce(tc)
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root); addpath(fullfile(root, 'config')); addpath(genpath(fullfile(root, 'src')));
cfg = getConfig();
base = fullfile(fileparts(cfg.datasetRoot), 'work', 'degraded');
if ~isfolder(base), base = tempdir; end
tc.TestData.out = fullfile(base, ['_unittest_r1_' char(java.util.UUID.randomUUID)]);
mkdir(tc.TestData.out);
J = jsondecode(fileread(fullfile(root, 'shared', 'patches.json')));
nP = numel(J.patches); fs = 29.5; dur = 40; n = round(dur * fs); hr0 = 72;
t = (0:n-1)' / fs;
rng(11);
A.t = t; A.names = string({J.patches.name}); A.groups = string({J.patches.group});
A.rgb = NaN(n, 3, nP); A.npix = 500 * ones(n, nP); A.valid = true(n, nP);
pulse = sin(2 * pi * hr0 / 60 * t);
for q = 1:nP
    A.rgb(:, :, q) = [170 120 100] .* (1 + 0.003 * pulse .* [0.4 1 0.6]) + 0.2 * randn(n, 3);
end
gap = t >= 12 & t < 18;                                  % 6 s without a face
A.valid(gap, :) = false; A.rgb(gap, :, :) = NaN; A.npix(gap, :) = 0;
occ = t >= 28; isC = A.groups == "C";                    % lower face masked from 28 s
A.valid(occ, isC) = false; A.rgb(occ, :, isC) = NaN; A.npix(occ, isC) = 0;
A.poly = repmat({[10 10; 20 10; 15 20]}, n, nP);
tc.TestData.A = A; tc.TestData.hr0 = hr0; tc.TestData.t = t;
tg = (0:1/60:dur)';
tc.TestData.ref = struct('t', tg, 'ppg', sin(2 * pi * hr0 / 60 * tg) + 0.05 * randn(size(tg)), ...
    'hrSensor', hr0 * ones(size(tg)));
% tiny synthetic "video" for facial_video
vp = fullfile(tc.TestData.out, 'synthetic.avi');
w = VideoWriter(vp, 'Uncompressed AVI'); w.FrameRate = fs; open(w);
for k = 1:60, writeVideo(w, uint8(100 + 20 * rand(24, 32, 3))); end
close(w);
tc.TestData.video = vp;
end

function teardownOnce(tc)
if isfolder(tc.TestData.out), rmdir(tc.TestData.out, 's'); end
end

function testEndToEnd(tc)
od = fullfile(tc.TestData.out, 'Round1_Baseline');
out = process_round1(tc.TestData.video, 'Reference', tc.TestData.ref, 'OutDir', od, ...
    'Constraint', 'None', 'Condition', 'Baseline', 'Traces', tc.TestData.A, ...
    'MaxFramesInMat', 20, 'Params', struct('lux', 300));
% CSV
T = readtable(fullfile(od, 'processed_output.csv'), 'TextType', 'string');
tc.verifyEqual(T.Properties.VariableNames, {'time_s', 'hr_bpm', 'confidence', 'status'});
tc.verifyClass(T.time_s, 'double'); tc.verifyClass(T.hr_bpm, 'double'); tc.verifyClass(T.status, 'string');
tc.verifyTrue(all(ismember(T.status, ["OK" "MASKED" "NO_FACE" "LOW_SNR" "LOW_COVERAGE"])));
tc.verifyEqual(diff(T.time_s), ones(height(T) - 1, 1), 'RelTol', 0.02);   % 1 s step = round(fs) samples
tc.verifyTrue(all(T.confidence >= 0 & T.confidence <= 1));
okm = T.status == "OK" | T.status == "MASKED";
tc.verifyTrue(all(isnan(T.hr_bpm(~okm))));                                % HR NaN unless OK/MASKED
tc.verifyLessThan(mean(abs(T.hr_bpm(okm) - tc.TestData.hr0)), 1.0);
% status logic: window centred in the gap -> NO_FACE; windows fully after 28 s -> MASKED
tc.verifyTrue(all(T.status(abs(T.time_s - 15) < 0.6) == "NO_FACE"));
late = T.time_s > 33.1;
tc.verifyTrue(all(T.status(late) == "MASKED"));
tc.verifyTrue(all(T.status(T.time_s < 7) == "OK"));
% MAT: exactly the six variables, types
mf = fullfile(od, 'Round1_Baseline_None.mat');
tc.verifyEqual(string(out.matFile), string(mf));
w = whos('-file', mf);
tc.verifyEqual(sort(string({w.name})), sort(["facial_video" "roi_data" "reference_signal" "processed_signal" "metrics" "params"]));
S = load(mf);
tc.verifyClass(S.facial_video, 'uint8'); tc.verifyEqual(size(S.facial_video), [24 32 3 20]);
tc.verifyTrue(all(isfield(S.processed_signal, {'t', 'bvp', 'hr_bpm', 'confidence', 'status'})));
tc.verifyTrue(all(isfield(S.reference_signal, {'t', 'ppg', 'hr_bpm', 'source'})));
tc.verifyTrue(all(isfield(S.roi_data, {'t', 'polygons', 'names', 'valid', 'weights'})));
tc.verifyEqual(size(S.roi_data.valid), [numel(tc.TestData.t) 10]);
tc.verifyTrue(all(isfield(S.metrics, {'MAE', 'RMSE', 'r', 'bias', 'loaLow', 'loaHigh', 'SUC3', 'SUC5', 'coverage'})));
tc.verifyLessThan(S.metrics.MAE, 1.0);
tc.verifyTrue(all(isfield(S.params, {'resolution', 'fps', 'lux', 'distance_cm', 'fitzpatrick', 'ita_deg', ...
    'age', 'constraints_applied', 'pipeline_version', 'git_hash'})));
tc.verifyEqual(S.params.lux, 300);
tc.verifyEqual(S.params.fps, 29.5, 'AbsTol', 1e-6);
end

function testNoReferenceWritesCsvOnly(tc)
od = fullfile(tc.TestData.out, 'noref');
out = process_round1(tc.TestData.video, 'OutDir', od, 'Traces', tc.TestData.A);
tc.verifyTrue(isfile(fullfile(od, 'processed_output.csv')));
tc.verifyEmpty(dir(fullfile(od, '*.mat')));
tc.verifyEmpty(out.matFile);
end

function testOmitVideoAndViterbi(tc)
od = fullfile(tc.TestData.out, 'viterbi');
out = process_round1(tc.TestData.video, 'Reference', tc.TestData.ref, 'OutDir', od, ...
    'Traces', tc.TestData.A, 'MaxFramesInMat', 0, 'Tracker', 'viterbi', ...
    'Condition', 'Masked', 'Constraint', 'C7');
S = load(fullfile(od, 'Round1_Masked_C7.mat'), 'facial_video', 'metrics');
tc.verifyClass(S.facial_video, 'uint8'); tc.verifyEqual(size(S.facial_video, 4), 0);
tc.verifyLessThan(S.metrics.MAE, 1.0);
tc.verifyEqual(out.params.tracker, 'viterbi');
end

function testReferenceCsvHrOnly(tc)
od = fullfile(tc.TestData.out, 'csvref');
mkdir(od);
rf = fullfile(od, 'reference_signal.csv');
g = tc.TestData.ref;
writetable(table(g.t, g.hrSensor, 'VariableNames', {'time_s', 'hr_bpm'}), rf);
out = process_round1(tc.TestData.video, 'Reference', rf, 'OutDir', od, 'Traces', tc.TestData.A, 'MaxFramesInMat', 0);
tc.verifyLessThan(out.metrics.MAE, 1.0);
tc.verifyTrue(contains(out.metrics.reference, 'no PPG'));
end
