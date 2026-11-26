function check_crop_motion()
%CHECK_CROP_MOTION Reliability of the phase-correlation crop motion (aipex_cache_dev3), AiPEX-dev.
d = dir("D:\SPCup Data\work\aipex_iter\traces3\*.mat");
v = zeros(0, 2);
for k = 1:numel(d)
    S = load(fullfile(d(k).folder, d(k).name), 'M');
    for r = ["forehead" "leftcheek" "rightcheek"]
        m = vecnorm(S.M.(r)(2:end, :), 2, 2);
        v(end+1, :) = [median(m) mean(m < 0.5)]; %#ok<AGROW>
    end
end
fprintf('crops %d: median inter-frame shift %.1f px (range %.1f-%.1f); share of frames < 0.5 px: median %.2f\n', ...
    size(v, 1), median(v(:, 1)), min(v(:, 1)), max(v(:, 1)), median(v(:, 2)));
end
