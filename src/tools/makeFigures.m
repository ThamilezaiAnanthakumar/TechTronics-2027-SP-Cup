function files = makeFigures(traces, R, ref, tRef, hrRef, best, cfg)
%MAKEFIGURES Save diagnostic PNGs to <resultsRoot>/<subject>/figures/.
%   files = makeFigures(traces, R, ref, tRef, hrRef, best, cfg)
%   traces : from extractRGBTraces;  R : struct array of results
%            (method, roi, pf, tHR, hr, spec, m);  ref : from referenceHR
%   best   : index into R of the best method/ROI (lowest MAE)
%   Figures are created invisible and closed after saving.
%   No face images are saved for cfg.privateSubjects (UBFC 21, 27).

figDir = fullfile(cfg.resultsRoot, cfg.subject, 'figures');
if ~isfolder(figDir), mkdir(figDir); end
files = {};
b = R(best);
bestRoi = b.roi;
label = sprintf('%s / %s', b.method, b.roi);

% 1. ROI overlay ---------------------------------------------------------
if isPrivateSubject(cfg)
    fprintf('makeFigures: %s is a privacy-restricted subject, no ROI overlay saved.\n', cfg.subject);
else
    fi = find(traces.faceOK, 1, 'first');
    fi = max(fi, min(traces.info.numFrames, round(5 * traces.fps)));   % ~5 s in
    v = VideoReader(traces.info.videoPath);
    v.CurrentTime = traces.t(fi);
    frame = degradeFrame(readFrame(v), cfg.degrade);
    box = traces.box(fi, :);
    rb = roiBoxes(box, cfg, size(frame, [1 2]));
    names = fieldnames(rb);
    rects = zeros(numel(names), 4);
    for k = 1:numel(names)
        q = rb.(names{k});
        rects(k, :) = [q(3) q(1) q(4)-q(3) q(2)-q(1)];
    end
    img = insertShape(frame, 'rectangle', box, 'Color', 'yellow', 'LineWidth', 2);
    img = insertShape(img, 'rectangle', rects, 'Color', 'green', 'LineWidth', 2);
    files{end+1} = fullfile(figDir, '1_roi_overlay.png');
    imwrite(img, files{end});
end

% 2. Raw RGB traces (combined ROI) --------------------------------------
fig = newFig();
rgb = traces.rgb.combined;
cols = {'r', 'g', 'b'}; chn = {'R', 'G', 'B'};
for c = 1:3
    subplot(3, 1, c);
    plot(traces.t, rgb(:, c), cols{c}); grid on;
    ylabel(chn{c});
    if c == 1, title('Raw mean RGB, combined ROI'); end
end
xlabel('Time (s)');
files{end+1} = saveFig(fig, figDir, '2_raw_rgb_combined.png');

% 3. Pulse vs contact PPG snippet ----------------------------------------
fig = newFig();
t0 = max(0, traces.t(end)/2 - cfg.snippetSec/2);
idx = traces.t >= t0 & traces.t < t0 + cfg.snippetSec;
s = sign(corr(b.pf, ref.ppg)); if s == 0, s = 1; end
h1 = plot(traces.t(idx), s * b.pf(idx), 'b-'); hold on; grid on;
h2 = plot(traces.t(idx), ref.ppg(idx), 'k-');
xlabel('Time (s)'); ylabel('z-score');
legend([h1 h2], sprintf('rPPG (%s)%s', label, ternary(s < 0, ', sign-flipped', '')), ...
    'Contact PPG', 'Location', 'best');
title(sprintf('%g s snippet: extracted pulse vs contact PPG', cfg.snippetSec));
files{end+1} = saveFig(fig, figDir, '3_pulse_vs_ppg.png');

% 4. Spectrum of one window ----------------------------------------------
fig = newFig();
w = ceil(numel(tRef) / 2);
bpm = 60 * b.spec.f;
pe = b.spec.P(w, :) / max(b.spec.P(w, :));
pr = ref.spec.P(w, :) / max(ref.spec.P(w, :));
plot(bpm, pe, 'b', bpm, pr, 'k'); grid on; hold on;
xline(b.hr(w), 'b--', sprintf('est %.1f', b.hr(w)));
xline(hrRef(w), 'k--', sprintf('ref %.1f', hrRef(w)));
xlabel('HR (bpm)'); ylabel('Normalised power');
legend(sprintf('rPPG (%s)', label), 'Contact PPG', 'Location', 'best');
title(sprintf('Spectrum of window centred at %.1f s', tRef(w)));
files{end+1} = saveFig(fig, figDir, '4_spectrum_window.png');

% 5. HR over time, each method on the best ROI ----------------------------
fig = newFig();
plot(tRef, hrRef, 'k-', 'LineWidth', 2); hold on; grid on;
plot(tRef, ref.hrSensor, '--', 'Color', [0.5 0.5 0.5]);
leg = {'Reference (contact PPG)', 'Oximeter HR'};
for k = 1:numel(R)
    if strcmp(R(k).roi, bestRoi)
        plot(R(k).tHR, R(k).hr, '.-');
        leg{end+1} = sprintf('%s (MAE %.1f)', R(k).method, R(k).m.MAE); %#ok<AGROW>
    end
end
xlabel('Time (s)'); ylabel('HR (bpm)');
legend(leg, 'Location', 'best');
title(sprintf('HR over time, ROI = %s', bestRoi));
files{end+1} = saveFig(fig, figDir, '5_hr_over_time.png');

% 6. Bland-Altman + scatter for the best method ---------------------------
fig = newFig([100 100 1100 450]);
m = b.m;
subplot(1, 2, 1);
avg = (b.hr + hrRef) / 2; d = b.hr - hrRef;
plot(avg, d, 'o'); grid on; hold on;
yline(m.bias, 'k-', sprintf('bias %.2f', m.bias));
yline(m.loaLow, 'r--', sprintf('%.2f', m.loaLow));
yline(m.loaHigh, 'r--', sprintf('%.2f', m.loaHigh));
xlabel('Mean of est. and ref. (bpm)'); ylabel('Est. - ref. (bpm)');
title(['Bland-Altman: ' label]);
subplot(1, 2, 2);
plot(hrRef, b.hr, 'o'); grid on; hold on;
lim = [min([hrRef; b.hr]) max([hrRef; b.hr])] + [-5 5];
plot(lim, lim, 'k--'); xlim(lim); ylim(lim); axis square;
xlabel('Reference HR (bpm)'); ylabel('Estimated HR (bpm)');
title(sprintf('r = %.3f, MAE = %.2f bpm', m.r, m.MAE));
files{end+1} = saveFig(fig, figDir, '6_bland_altman_scatter.png');
end

% ------------------------------------------------------------------------
function fig = newFig(pos)
if nargin < 1, pos = [100 100 900 500]; end
fig = figure('Visible', 'off', 'Position', pos);
end

function f = saveFig(fig, figDir, name)
f = fullfile(figDir, name);
exportgraphics(fig, f, 'Resolution', 150);
close(fig);
end

function tf = isPrivateSubject(cfg)
n = str2double(regexp(cfg.subject, '\d+', 'match', 'once'));
tf = ~isnan(n) && ismember(n, cfg.privateSubjects);
end

function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end
