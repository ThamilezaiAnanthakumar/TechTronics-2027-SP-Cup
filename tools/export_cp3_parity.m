function export_cp3_parity(outDir, subjects, durS)
%EXPORT_CP3_PARITY Export CP3 parity fixtures for the FaceVitals web core (web/src/core/cp3Core.ts).
%   export_cp3_parity()                       -> ../wt_apps/web/tests/fixtures, subjects 1 3 12, 60 s
%   export_cp3_parity(outDir, [1 3 12], 60)
%   For each UBFC DEV subject: the cached anatomical-patch traces (D:\SPCup Data\work\anat_traces)
%   cut to the first durS seconds and rounded to 4 decimals, then src/pipeline/snrFusionHR.m run on
%   exactly those rounded numbers (so the TS port sees the same input). Writes
%   cp3_parity_subject<k>.json with t, r/g/b/npix/valid (N x 10), MATLAB window centres tc + hr,
%   and the butter(3, [0.7 3]/15, 'bandpass') coefficients b, a.
%   Only per-frame mean RGB numbers are exported (no images). DEV subjects only.
arguments
    outDir (1,1) string = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), "wt_apps", "web", "tests", "fixtures")
    subjects (1,:) double = [1 3 12]
    durS (1,1) double = 60
end
dev = [1 3 5 8 9 10 11 12 13 14 15 16];
assert(all(ismember(subjects, dev)), 'export_cp3_parity: DEV subjects only');
root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root, 'src')));
if ~isfolder(outDir), mkdir(outDir); end
rnd = @(x) round(x * 1e4) / 1e4;
[b, a] = butter(3, [0.7 3.0] / 15, 'bandpass');
for s = subjects
    S = load(fullfile("D:\SPCup Data\work\anat_traces", "subject" + s + ".mat"), 'A'); A = S.A;
    keep = A.t - A.t(1) <= durS;
    t = rnd(A.t(keep)); nP = size(A.rgb, 3);
    rgb = rnd(A.rgb(keep, :, :)); npix = double(A.npix(keep, :)); valid = logical(A.valid(keep, :));
    X = cell(1, nP); W = cell(1, nP);
    for q = 1:nP
        x = rgb(:, :, q); x(~valid(:, q), :) = NaN; X{q} = x; W{q} = npix(:, q);
    end
    [hr, tc] = snrFusionHR(t, X, W);
    J = struct('subject', "subject" + s, 'duration_s', durS, 'source', ...
        "tools/export_cp3_parity.m -> src/pipeline/snrFusionHR.m (defaults = CP3 freeze)", ...
        't', t, 'r', squeeze(rgb(:, 1, :)), 'g', squeeze(rgb(:, 2, :)), 'b', squeeze(rgb(:, 3, :)), ...
        'npix', npix, 'valid', double(valid), 'tc', tc, 'hr', hr, 'butter_b', b, 'butter_a', a);
    f = fullfile(outDir, "cp3_parity_subject" + s + ".json");
    fid = fopen(f, 'w'); fwrite(fid, jsonencode(J, 'ConvertInfAndNaN', true)); fclose(fid);
    d = dir(f);
    fprintf('subject%d: %d frames, %d windows (%d with HR), %.0f kB -> %s\n', s, numel(t), numel(hr), ...
        nnz(isfinite(hr)), d.bytes / 1024, f);
end
end
