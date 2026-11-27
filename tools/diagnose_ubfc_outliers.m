function T = diagnose_ubfc_outliers(subjects, opts)
%DIAGNOSE_UBFC_OUTLIERS Failure analysis of UBFC2 outlier subjects (README 2.2). Analysis only, no tuning.
%   T = diagnose_ubfc_outliers()                 subjects 15, 49, 27, 41 (+ 1, 12 as good controls)
%   For each subject, from the seeded run_ubfc_all outputs (traces.mat, hr_results.mat):
%     - per-window error of POS/combined vs the PPG reference
%     - per-ROI window SNR (power within +-6 bpm of the peak and its harmonic / rest of the band)
%     - face-box motion (median and 95th percentile of |d centre| per frame, px) and scale change
%     - share of windows whose peak is near 2x or 0.5x the reference (harmonic lock)
%     - share of windows whose rPPG peak lies within 3 bpm of the dominant motion frequency
%   Saves outputs/diagnose_ubfc/<subject>.png (spectrogram with reference track, per-ROI SNR,
%   motion) and outputs/diagnose_ubfc/summary.csv. Figures contain no face images.
arguments
    subjects (1,:) double = [15 49 27 41 1 12]
    opts.ResultsRoot (1,1) string = "D:\SPCup Data\work\ubfc2_seeded"
end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'config')); addpath(genpath(fullfile(root, 'src')));
cfg = getConfig();
outDir = fullfile(root, 'outputs', 'diagnose_ubfc');
if ~isfolder(outDir), mkdir(outDir); end
rows = {};
for s = subjects
    subj = "subject" + s;
    L = load(fullfile(opts.ResultsRoot, subj, 'traces.mat'), 'traces'); tr = L.traces;
    H = load(fullfile(opts.ResultsRoot, subj, 'hr_results.mat'), 'R', 'hrRef', 'tRef', 'ref');
    i = find(strcmp({H.R.method}, 'POS') & strcmp({H.R.roi}, 'combined'), 1);
    Rb = H.R(i);
    err = Rb.hr - H.hrRef;
    ratio = Rb.hr ./ H.hrRef;
    harm = mean(abs(ratio - 2) < 0.15 | abs(ratio - 0.5) < 0.08);

    % per-ROI SNR per window (same chain as run_pipeline)
    rois = cfg.roiSet; snr = NaN(numel(Rb.hr), numel(rois));
    for r = 1:numel(rois)
        k = find(strcmp({H.R.method}, 'POS') & strcmp({H.R.roi}, rois{r}), 1);
        P = H.R(k).spec.P; f = H.R(k).spec.f;
        for w = 1:size(P, 1)
            [~, j] = max(P(w, :)); f0 = f(j);
            sig = abs(f - f0) <= 0.1 | abs(f - 2 * f0) <= 0.1;
            snr(w, r) = 10 * log10(sum(P(w, sig)) / max(sum(P(w, ~sig)), eps));
        end
    end

    % face-box motion
    c = tr.box(:, 1:2) + tr.box(:, 3:4) / 2;
    dc = vecnorm(diff(c), 2, 2);
    sc = abs(diff(log(tr.box(:, 3))));
    % dominant motion frequency per window (box centre x/y), in the HR band
    L = Rb.spec.winLen; st = Rb.spec.starts; fsF = tr.fps;
    motionLock = false(numel(st), 1);
    for w = 1:numel(st)
        seg = c(st(w):st(w)+L-1, :); seg = seg - mean(seg, 1);
        nfft = 2^14; F = (0:nfft/2)' * fsF / nfft; band = F >= cfg.hrBand(1) & F <= cfg.hrBand(2);
        Pm = sum(abs(fft(seg .* hann(L), nfft)).^2, 2); Pm = Pm(1:nfft/2+1);
        [~, j] = max(Pm(band)); Fb = F(band);
        motionLock(w) = abs(60 * Fb(j) - Rb.hr(w)) < 3 && abs(err(w)) > 5;
    end

    rows(end+1, :) = {subj, mean(abs(err)), median(abs(err)), mean(abs(err) > 5) * 100, harm * 100, ...
        mean(motionLock) * 100, median(snr(:, end)), median(snr(:, 1)), median(snr(:, 2)), median(snr(:, 3)), ...
        median(dc), prctile(dc, 95), prctile(sc, 95) * 100, mean(tr.faceOK) * 100, ...
        median(H.hrRef), max(H.hrRef) - min(H.hrRef)}; %#ok<AGROW>

    % figure (no images of the face)
    f = figure('Visible', 'off', 'Position', [0 0 1100 650]);
    if exist('theme', 'file'), theme(f, 'light'); end
    subplot(3, 1, 1);
    imagesc(Rb.tHR, 60 * Rb.spec.f, (Rb.spec.P ./ max(Rb.spec.P, [], 2))'); axis xy; hold on;
    plot(H.tRef, H.hrRef, 'w-', 'LineWidth', 1.5); plot(Rb.tHR, Rb.hr, 'r.', 'MarkerSize', 6);
    ylabel('bpm'); title(sprintf('%s POS/combined: MAE %.2f, |err|>5 in %.0f%% of windows (white = PPG ref, red = estimate)', ...
        subj, mean(abs(err)), 100 * mean(abs(err) > 5)));
    subplot(3, 1, 2);
    plot(Rb.tHR, snr); legend(rois, 'Location', 'eastoutside'); ylabel('SNR (dB)'); grid on;
    subplot(3, 1, 3);
    yyaxis left; plot(tr.t(2:end), dc); ylabel('|\Delta centre| (px/frame)');
    yyaxis right; plot(Rb.tHR, abs(err), '.'); ylabel('|error| (bpm)'); xlabel('time (s)'); grid on;
    exportgraphics(f, fullfile(outDir, subj + ".png")); close(f);
end
T = cell2table(rows, 'VariableNames', {'Subject', 'MAE', 'MedianAE', 'PctErrGt5', 'PctHarmonic', ...
    'PctMotionLock', 'SNRcombined_dB', 'SNRforehead_dB', 'SNRleft_dB', 'SNRright_dB', ...
    'MotionMedian_px', 'Motion95_px', 'Scale95_pct', 'FaceOK_pct', 'RefMedian', 'RefRange'});
writetable(T, fullfile(outDir, 'summary.csv'));
disp(T);
end
