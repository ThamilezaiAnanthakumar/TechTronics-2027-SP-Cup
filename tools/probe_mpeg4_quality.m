%PROBE_MPEG4_QUALITY Does VideoWriter 'MPEG-4' Quality change the bitrate? (60 frames of UBFC dev subject 1)
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'config')); addpath(genpath(fullfile(root, 'src')));
cfg = getConfig('subject', 'subject1'); [vp, ~] = loadUBFC(cfg);
v = VideoReader(vp); F = read(v, [1 60]);
out = 'D:\SPCup Data\work\probe_mpeg4'; if ~isfolder(out), mkdir(out); end
for q = [25 50 75 95 100]
    f = fullfile(out, sprintf('q%d.mp4', q));
    w = VideoWriter(f, 'MPEG-4'); w.FrameRate = 30; w.Quality = q; open(w); writeVideo(w, F); close(w);
    r = VideoReader(f); G = read(r, [1 60]);
    d = double(G) - double(F);
    fprintf('Quality %3d: %5.0f kbps, mean |error| %.2f grey levels, max %d\n', q, ...
        dir(f).bytes * 8 / 1000 / (60 / 30), mean(abs(d(:))), max(abs(d(:))));
end
