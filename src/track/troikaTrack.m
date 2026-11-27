function [hr, info] = troikaTrack(P, f, opts)
%TROIKATRACK TROIKA-style spectral peak tracking over a sequence of window spectra.
%   [hr, info] = troikaTrack(P, f)
%   [hr, info] = troikaTrack(P, f, Step=1, JumpBpm=5, StepBpm=2, SearchBpm=15, ...)
%   P   nWin x nf  window spectra (e.g. the fused spectra of snrFusionHR); NaN rows = no estimate
%   f   nf x 1     frequencies of P (Hz)
%   hr  nWin x 1   tracked HR (bpm); NaN until the first valid window
%   info.action    per window: "init" | "harmonic" | "nearest" | "hold" | "clamp" | "trend" | "gap"
%
%   Follows Zhang, Pi & Liu, TROIKA, IEEE TBME 62(2), 2015, Section III-D (arXiv:1409.5181),
%   with every parameter in physical units (README_UPDATE_01 U3) instead of FFT bins at 125 Hz:
%     Initialisation      highest spectral peak of the first valid window (TROIKA: highest peak
%                         during a still period).
%     Peak selection      search range R0 = prev +- SearchBpm and R1 = 2x that range (first
%                         harmonic); keep up to 3 highest local peaks >= PeakRel x max(R0) in each.
%                         Harmonic pair (|f1 - 2 f0| <= HarmTolBpm) -> that fundamental (Eq. 13);
%                         otherwise the candidate (R0 peaks and halved R1 peaks) closest to prev
%                         (Eq. 14); no peaks -> prev (Eq. 15).
%     Verification        if |est - prev| >= JumpBpm*Step: est = prev +- StepBpm*Step (Eq. 16), so a
%                         reading never jumps by JumpBpm or more between consecutive windows.
%     Lost tracking       if est == prev for LostWindows windows: est = prev + 2*trend*StepBpm, with
%                         trend = sign of (cubic fit over the last TrendWindows estimates predicted one
%                         step ahead - prev) when |.| >= TrendBpm (Eqs. 17-18); search range widened
%                         to SearchLostBpm while lost.
%     Gaps                after more than GapResetS seconds without a valid window the tracker
%                         re-initialises (not in TROIKA: the face can leave the camera view).
%   TROIKA's values at 125 Hz, 2 s step, 4096-point FFT: Delta_s 16 bins (~29 bpm), theta 6 bins
%   (~11 bpm), tau 2 bins (~3.7 bpm), eta 30 %, h 3, 20 windows, trend threshold 3 bpm. Defaults
%   here: JumpBpm 5 (operator requirement: no change of 5 bpm or more between readings),
%   StepBpm 2 (TROIKA tau/theta ratio ~1/3), SearchBpm 15 (half of TROIKA's range for a 1 s step).
arguments
    P double
    f (:,1) double
    opts.Step (1,1) double = 1            % seconds between windows
    opts.JumpBpm (1,1) double = 5         % per second
    opts.StepBpm (1,1) double = 2         % per second
    opts.SearchBpm (1,1) double = 15
    opts.SearchLostBpm (1,1) double = 19
    opts.PeakRel (1,1) double = 0.3
    opts.HarmTolBpm (1,1) double = 3
    opts.LostWindows (1,1) double = 3
    opts.TrendWindows (1,1) double = 20
    opts.TrendBpm (1,1) double = 3
    opts.GapResetS (1,1) double = 10
    opts.InitWindows (1,1) double = 1      % consensus start (5 tried on UBFC dev: no gain): median of the first raw peaks (not in TROIKA)
    opts.RelockWindows (1,1) double = Inf  % sustained (5 tried on UBFC dev: no gain), self-consistent disagreement -> re-lock (not in TROIKA)
    opts.RelockStepBpm (1,1) double = 4.5  % per second; stays below JumpBpm
end
bpm = 60 * f(:);
nW = size(P, 1);
hr = NaN(nW, 1); action = strings(nW, 1);
prev = NaN; nSame = 0; lastValid = -Inf; hist = NaN(nW, 1);
rawBuf = []; nDis = 0; rawHist = NaN(nW, 1);
jump = opts.JumpBpm * opts.Step; tau = opts.StepBpm * opts.Step;
for w = 1:nW
    p = P(w, :)';
    if all(~isfinite(p))
        action(w) = "gap";
        if isfinite(prev) && (w - lastValid) * opts.Step > opts.GapResetS, prev = NaN; nSame = 0; end
        continue
    end
    p(~isfinite(p)) = 0;
    [~, kr] = max(p); raw = bpm(kr); rawHist(w) = raw;
    if ~isfinite(prev)                                   % (re-)initialisation: consensus of the first raw peaks
        rawBuf(end+1) = raw; %#ok<AGROW>
        hr(w) = median(rawBuf); action(w) = "init"; lastValid = w;
        if numel(rawBuf) >= opts.InitWindows, prev = median(rawBuf); hist(w) = prev; rawBuf = []; nDis = 0; end
        continue
    else
        lost = nSame >= opts.LostWindows;
        half = opts.SearchBpm; if lost, half = opts.SearchLostBpm; end
        r0 = abs(bpm - prev) <= half;
        r1 = bpm >= 2 * (prev - half) & bpm <= 2 * (prev + half);
        thr = Inf; if any(r0), thr = opts.PeakRel * max(p(r0)); end
        c0 = topPeaks(p, r0, thr); c1 = topPeaks(p, r1, thr);
        est = NaN;
        for i = 1:numel(c0)                              % Eq. 13: harmonic pair
            if any(abs(bpm(c1) - 2 * bpm(c0(i))) <= opts.HarmTolBpm), est = bpm(c0(i)); action(w) = "harmonic"; break; end
        end
        if isnan(est)
            cand = [bpm(c0); bpm(c1) / 2];               % Eq. 14: nearest to the previous estimate
            if ~isempty(cand)
                [~, j] = min(abs(cand - prev)); est = cand(j); action(w) = "nearest";
            else
                est = prev; action(w) = "hold";          % Eq. 15
            end
        end
        if abs(est - prev) >= jump                       % Eq. 16: verification / clamp
            est = prev + sign(est - prev) * tau; action(w) = "clamp";
        end
        % re-lock: the raw peak has disagreed with the track by >= jump for RelockWindows windows and the
        % raw peaks agree with each other (spread < jump) -> catch up at RelockStepBpm per step (< JumpBpm)
        if abs(raw - prev) >= jump, nDis = nDis + 1; else, nDis = 0; end
        if nDis >= opts.RelockWindows
            rr = rawHist(w - opts.RelockWindows + 1:w);
            if all(isfinite(rr)) && (max(rr) - min(rr)) < jump
                tgt = median(rr); stepR = min(opts.RelockStepBpm * opts.Step, abs(tgt - prev));
                est = prev + sign(tgt - prev) * stepR; action(w) = "relock";
            end
        end
        if est == prev, nSame = nSame + 1; else, nSame = 0; end
        if nSame >= opts.LostWindows                     % Eqs. 17-18: trend step when stuck
            h = hist(max(1, w - opts.TrendWindows):w - 1); h = h(isfinite(h));
            if numel(h) >= 4
                x = (1:numel(h))'; c = polyfit(x, h, 3); pred = polyval(c, numel(h) + 1);
                if pred - prev >= opts.TrendBpm, est = prev + 2 * tau; action(w) = "trend";
                elseif pred - prev <= -opts.TrendBpm, est = prev - 2 * tau; action(w) = "trend"; end
            end
        end
    end
    hr(w) = est; hist(w) = est; prev = est; lastValid = w;
end
info = struct('action', action, 'params', opts);
end

function idx = topPeaks(p, mask, thr)
% up to three highest local maxima of p inside mask with p >= thr
idx = find(mask);
if numel(idx) < 3 || ~isfinite(thr), idx = zeros(0, 1); return; end
q = p(idx);
isPk = [false; q(2:end-1) > q(1:end-2) & q(2:end-1) >= q(3:end); false] & q >= thr;
idx = idx(isPk);
[~, o] = sort(p(idx), 'descend'); idx = idx(o(1:min(3, numel(o))));
end
