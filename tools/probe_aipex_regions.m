function T = probe_aipex_regions()
%PROBE_AIPEX_REGIONS Frame counts / header fps / size of the three ROI crops per AiPEX-dev video.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'config')); cfg = getConfig();
ids = readlines(fullfile(root, 'shared', 'splits', 'aipex_dev.txt'), 'EmptyLineRule', 'skip');
reg = ["forehead" "leftcheek" "rightcheek"];
rows = cell(numel(ids), 2 + 3*3);
for i = 1:numel(ids)
    p = split(ids(i), '/'); rows{i,1} = p(1); rows{i,2} = p(2);
    for r = 1:3
        v = VideoReader(fullfile(cfg.aipexRoot, 'Data', p(1), p(2) + "_" + reg(r) + ".avi"));
        rows{i, 2+r} = v.NumFrames; rows{i, 5+r} = v.FrameRate; rows{i, 8+r} = sprintf('%dx%d', v.Width, v.Height);
    end
end
T = cell2table(rows, 'VariableNames', ['Country' 'Video' "N_" + reg "fps_" + reg "size_" + reg]);
writetable(T, fullfile(root, 'outputs', 'aipex_iter', 'regions_dev.csv'));
same = T.N_forehead == T.N_leftcheek & T.N_leftcheek == T.N_rightcheek;
fprintf('dev videos with equal frame counts in all 3 crops: %d / %d\n', nnz(same), height(T));
ratio = T.N_forehead ./ min(T.N_leftcheek, T.N_rightcheek);
fprintf('forehead/cheek frame-count ratio: min %.2f median %.2f max %.2f\n', min(ratio), median(ratio), max(ratio));
fprintf('sizes: %s | %s | %s\n', strjoin(unique(string(T.size_forehead))', ','), strjoin(unique(string(T.size_leftcheek))', ','), strjoin(unique(string(T.size_rightcheek))', ','));
disp(T(1:6, 1:5));
end
