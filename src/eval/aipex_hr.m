function [T, S] = aipex_hr(opts)
%AIPEX_HR  HR on the AiPEX rppg_biases clips: face_detect baseline vs + Kalman denoise.
%   [T, S] = aipex_hr()
%   [T, S] = aipex_hr(Root=..., OutDir=..., TraceDir=..., Win=10, Rerun=false)
%
%   AiPEX gives pre-cropped skin clips (forehead / left cheek / right cheek)
%   and one mean reference HR per video, so face detection and tracking are
%   skipped. The front end is the face_detect one, unchanged:
%     skin_mask (face_params skin box) -> mean RGB of skin pixels per region
%     -> rgb_face = pixel-weighted mean of the regions
%   Methods compared on that same rgb_face:
%     pos        face_detect sanity_hr (POS, 0.7-3 Hz, 10 s windows every
%                1 s, strongest periodogram peak)         <- baseline
%     kf_denoise Kalman denoising of rgb_face (kalman_denoise, pixel-count
%                weighted), then the same unchanged sanity_hr
%   Video HR = mean of the window HRs, compared with the reference.
%
%   Per-frame region traces are cached in TraceDir, so reruns skip video
%   decoding (Rerun=true forces it). Also reports per-region baseline HR
%   and the Individual Typology Angle (ITA) of the skin pixels.
%
%   Writes  OutDir/aipex_hr_per_video.csv  (one row per video)
%           OutDir/aipex_hr_summary.csv    (per method and group: MAE, RMSE,
%                                           r, bias, LoA, SUC)

arguments
    opts.Root (1,1) string = fullfile(repo_root(), "Data", "AiPEX-Lab")
    opts.OutDir (1,1) string = fullfile(repo_root(), "outputs", "tables", "aipex")
    opts.TraceDir (1,1) string = fullfile(repo_root(), "outputs", "traces", "aipex")
    opts.Win (1,1) double = 10
    opts.Rerun (1,1) logical = false
    opts.MaxVideos (1,1) double = Inf    % per country (quick tests)
end

p = face_params();
regions   = ["forehead" "leftcheek" "rightcheek"];
countries = ["India" "Sierra_Leone"];
if ~isfolder(opts.TraceDir), mkdir(opts.TraceDir); end

rows = {};
for c = countries
    gt  = readtable(fullfile(opts.Root, "Ground_Truth", c + "_GT.csv"), ...
                    TextType="string", VariableNamingRule="preserve");
    ids = strtrim(gt{:, 1});
    ref = gt{:, 2};
    for i = 1:min(numel(ids), opts.MaxVideos)
        tic;
        tracePath = fullfile(opts.TraceDir, c + "_" + ids(i) + ".mat");
        if isfile(tracePath) && ~opts.Rerun
            L0 = load(tracePath);  R = L0.R;
        else
            R = struct();
            for r = regions
                R.(r) = read_region(fullfile(opts.Root, "Data", c, ids(i) + "_" + r + ".avi"), p);
            end
            save(tracePath, "R", "-v7");
        end

        % common length (region clips are recorded together)
        N = min(arrayfun(@(r) numel(R.(r).t), regions));
        t = R.forehead.t(1:N);
        num = zeros(N, 3);  den = zeros(N, 1);
        hrReg = NaN(1, numel(regions));
        for k = 1:numel(regions)
            Q  = R.(regions(k));
            w  = Q.npix(1:N);  x = Q.rgb(1:N, :);
            ok = isfinite(x(:, 1));
            num(ok, :) = num(ok, :) + w(ok) .* x(ok, :);
            den(ok)    = den(ok) + w(ok);
            hrReg(k)   = mean(sanity_hr(t, x, Win=opts.Win), "omitnan");
        end
        rgbFace = num ./ den;               % NaN where no region has skin

        % baseline: face_detect, unchanged
        hrPos = sanity_hr(t, rgbFace, Win=opts.Win);
        % Kalman denoising of rgb_face, then the same unchanged sanity_hr
        hrDn  = sanity_hr(t, kalman_denoise(rgbFace, R.forehead.fps, Npix=den), Win=opts.Win);

        meanRGB = mean(rgbFace, 1, "omitnan");
        lab = rgb2lab(meanRGB / 255);
        ita = atand((lab(1) - 50) / lab(3));
        skinRatio = mean(arrayfun(@(r) mean(R.(r).ratio, "omitnan"), regions));

        est = [mean(hrPos, "omitnan"), mean(hrDn, "omitnan")];
        rows(end+1, :) = {c, ids(i), ref(i), est(1), median(hrPos, "omitnan"), est(2), ...
            est(1) - ref(i), est(2) - ref(i), hrReg(1), hrReg(2), hrReg(3), numel(hrPos), ...
            t(end) - t(1), R.forehead.fps, skinRatio, lab(1), lab(3), ita, ita_class(ita)}; %#ok<AGROW>
        fprintf("%-12s %-8s ref %6.1f | pos %6.1f  kf_denoise %6.1f (%.1f s)\n", ...
            c, ids(i), ref(i), est(1), est(2), toc);
    end
end

T = cell2table(rows, VariableNames=["dataset" "video" "hr_ref" "hr_pos" "hr_pos_median" ...
    "hr_kf_denoise" "err_pos" "err_kf_denoise" "hr_forehead" "hr_lcheek" "hr_rcheek" ...
    "n_windows" "duration_s" "fps" "skin_ratio" "L" "b" "ITA" "ITA_class"]);

S = table();
for m = ["pos" "kf_denoise"]
    S = [S; summarize(T, m, "All")]; %#ok<AGROW>
    for c = countries
        S = [S; summarize(T(T.dataset == c, :), m, c)]; %#ok<AGROW>
    end
    for g = unique(T.ITA_class).'
        S = [S; summarize(T(T.ITA_class == g, :), m, "ITA: " + g)]; %#ok<AGROW>
    end
end

if ~isfolder(opts.OutDir), mkdir(opts.OutDir); end
writetable(T, fullfile(opts.OutDir, "aipex_hr_per_video.csv"));
writetable(S, fullfile(opts.OutDir, "aipex_hr_summary.csv"));
disp(S(S.group == "All" | S.group == "India" | S.group == "Sierra_Leone", :))
end

function Q = read_region(path, p)
% Mean RGB of skin pixels per frame, timestamps read from the video.
v = VideoReader(path);
n = v.NumFrames;
Q.t = zeros(n, 1);  Q.rgb = NaN(n, 3);  Q.npix = zeros(n, 1);  Q.ratio = NaN(n, 1);
Q.fps = v.FrameRate;
k = 0;
while hasFrame(v)
    k = k + 1;
    Q.t(k) = v.CurrentTime;
    P = readFrame(v);
    M = skin_mask(P, p);
    m = nnz(M);
    Q.npix(k)  = m;
    Q.ratio(k) = m / numel(M);
    if m >= p.occl.minPix
        P = reshape(double(P), [], 3);
        Q.rgb(k, :) = mean(P(M(:), :), 1);
    end
end
Q.t = Q.t(1:k);  Q.rgb = Q.rgb(1:k, :);  Q.npix = Q.npix(1:k);  Q.ratio = Q.ratio(1:k);
end

function s = ita_class(ita)
% Chardon et al. ITA categories (rough Fitzpatrick correspondence).
if     ita > 55,  s = "Very light (I)";
elseif ita > 41,  s = "Light (II)";
elseif ita > 28,  s = "Intermediate (III)";
elseif ita > 10,  s = "Tan (IV)";
elseif ita > -30, s = "Brown (V)";
else,             s = "Dark (VI)";
end
end

function S = summarize(T, method, name)
est = T.("hr_" + method);  ref = T.hr_ref;
e  = est - ref;  e = e(isfinite(e));
ok = isfinite(est) & isfinite(ref);
if nnz(ok) > 2, r = corr(est(ok), ref(ok)); else, r = NaN; end
bias = mean(e);  sd = std(e);
S = table(string(method), string(name), height(T), numel(e), mean(abs(e)), sqrt(mean(e.^2)), r, ...
    bias, bias - 1.96*sd, bias + 1.96*sd, 100*mean(abs(e) < 3), 100*mean(abs(e) < 5), ...
    100*mean(abs(e) < 10), mean(T.ITA), ...
    VariableNames=["method" "group" "n_videos" "n_valid" "MAE" "RMSE" "r" "bias" "LoA_low" ...
    "LoA_high" "SUC3" "SUC5" "SUC10" "mean_ITA"]);
end
