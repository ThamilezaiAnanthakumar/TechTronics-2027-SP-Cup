function export_troika_parity(outFile, trackerDir)
%EXPORT_TROIKA_PARITY Export troikaTrack.m outputs on deterministic synthetic spectra for the web port.
%   export_troika_parity()   -> ../wt_apps/web/tests/fixtures/troika_parity.json
%   export_troika_parity(outFile, trackerDir)   use the troikaTrack.m in trackerDir (e.g. a
%                            `git show 8f93c0f:src/track/troikaTrack.m` copy) instead of src/track
%   The spectra are NOT exported: they are generated identically in MATLAB and in
%   web/tests/troikaTrack.test.ts (MATLAB rng(seed) + rand = MT19937 init_genrand + genrand_res53,
%   reimplemented in web/tests/mt19937.ts), so the fixture holds only MATLAB's tracked HR and actions.
%   Scenarios (synthetic only):
%     motion  test_troika_track.m: 55 bpm, rise to 70 at 0.5 bpm/s, motion peaks 85 (every 7th)
%             and 65 (every 11th) 1.5x stronger; f = 0.7:0.0037:3.0; noise 0.05 rand, rng(5) (identical input)
%     hard    CP3 in-band bins (k*30/8192); 70 -> 100 (1.2 bpm/s) -> 65 (2.5 bpm/s, faster than the
%             clamp); random motion peaks 60-130 bpm (2x), missing harmonic every 5th window, pulse
%             absent in windows 120-127, gaps 60-63 and 150-161 (re-init); noise 0.3
%     step2   'motion' sampled every 2 s (Step = 2)
%   Also exports src/signal/viterbiHR.m (MaxDown = MaxUp = 4.99 bpm/s, offline full Viterbi) on
%   the same spectra (gap rows -> ones, as runOffline.m) for web/src/core/viterbiTrack.ts.
arguments
    outFile (1,1) string = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
        "wt_apps", "web", "tests", "fixtures", "troika_parity.json")
    trackerDir (1,1) string = ""
end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root, 'src')));
if trackerDir ~= "", addpath(trackerDir); end
fprintf('troikaTrack: %s\n', which('troikaTrack'));
S = struct('name', {}, 'step', {}, 'hr', {}, 'action', {}, 'hrTrue', {}, 'viterbi', {});
for name = ["motion", "hard", "step2"]
    [P, f, hrTrue, step] = scenario(name);
    [hr, info] = troikaTrack(P, f, Step=step);
    Pv = P; Pv(all(~isfinite(Pv), 2), :) = 1;       % no spectrum -> flat (as runOffline.m)
    hv = viterbiHR(Pv, f, Step=step, MaxDown=4.99, MaxUp=4.99);
    S(end + 1) = struct('name', name, 'step', step, 'hr', hr, 'action', info.action, 'hrTrue', hrTrue, 'viterbi', hv); %#ok<AGROW>
    fprintf('%-6s viterbi(4.99): max|err| %.2f, max step %.2f\n', name, max(abs(hv - hrTrue)), max(abs(diff(hv))));
    [~, k] = max(P, [], 2); am = 60 * f(k); am(all(~isfinite(P), 2)) = NaN;
    fprintf('%-6s argmax: max|err| %.2f, max step %.2f | troika: max|err| %.2f, max step %.2f | %s\n', name, ...
        max(abs(am - hrTrue)), max(abs(diff(am))), max(abs(hr - hrTrue)), max(abs(diff(hr))), ...
        strjoin(unique(info.action)', ','));
end
J = struct('source', "tools/export_troika_parity.m -> src/track/troikaTrack.m (defaults)", 'scenarios', S);
fid = fopen(outFile, 'w'); fwrite(fid, jsonencode(J, 'ConvertInfAndNaN', true)); fclose(fid);
fprintf('-> %s\n', outFile);
end

function [P, f, hrTrue, step] = scenario(name)
step = 1;
switch name
    case {"motion", "step2"}
        f = (0.7:0.0037:3.0)'; nW = 120;
        hrTrue = [55 * ones(40, 1); (55:0.5:70)'; 70 * ones(nW - 71, 1)];
        g = @(c) exp(-0.5 * ((60 * f' - c) / 1.5).^2);
        P = zeros(nW, numel(f)); rng(5);
        for w = 1:nW
            P(w, :) = g(hrTrue(w)) + 0.4 * g(2 * hrTrue(w)) + 0.05 * rand(1, numel(f));
            if mod(w, 7) == 0, P(w, :) = P(w, :) + 1.5 * g(85); end
            if mod(w, 11) == 0, P(w, :) = P(w, :) + 1.5 * g(65); end
        end
        if name == "step2"
            P = P(1:2:end, :); hrTrue = hrTrue(1:2:end); step = 2;
        end
    case "hard"
        k = (ceil(0.7 * 8192 / 30):floor(3.0 * 8192 / 30))';
        f = k * 30 / 8192; nW = 200;
        hrTrue = [70 * ones(30, 1); (70:1.2:100)'; 100 * ones(30, 1); (100:-2.5:65)'];
        hrTrue = [hrTrue; 65 * ones(nW - numel(hrTrue), 1)];
        g = @(c) exp(-0.5 * ((60 * f' - c) / 1.5).^2);
        P = zeros(nW, numel(f)); rng(7);
        for w = 1:nW
            a = rand(); b = rand();
            p = 0.3 * rand(1, numel(f));
            if w < 120 || w > 127
                p = p + g(hrTrue(w));
                if mod(w, 5) ~= 0, p = p + 0.4 * g(2 * hrTrue(w)); end
            end
            if a < 0.2, p = p + 2 * g(60 + 70 * b); end
            P(w, :) = p;
        end
        P([60:63, 150:161], :) = NaN;
end
end
