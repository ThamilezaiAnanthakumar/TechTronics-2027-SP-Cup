function T = probe_ffmpeg_mp4()
%PROBE_FFMPEG_MP4 Which ffmpeg H.264 MP4 keeps the pulse and is readable by MATLAB VideoReader?
%   30 s of UBFC dev subject 1 -> several encodings -> runOffline (frozen CP2) vs contact PPG.
%   Records size, bitrate, readability, MAE and coverage (docs/API_NOTES.md).
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root); addpath(fullfile(root, 'config')); addpath(genpath(fullfile(root, 'src')));
ff = fullfile(getenv('LOCALAPPDATA'), 'Microsoft', 'WinGet', 'Packages', ...
    'Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe', 'ffmpeg-9.1-full_build', 'bin', 'ffmpeg.exe');
cfg = getConfig('subject', 'subject1'); [vp, gt] = loadUBFC(cfg);
out = "D:\SPCup Data\work\probe_ffmpeg"; if ~isfolder(out), mkdir(out); end
V = { "x264_crf4_aq0",      "-c:v libx264 -preset slow -crf 4 -pix_fmt yuv420p -x264-params aq-mode=0:deblock=0,0:psy-rd=0,0 -tune grain";
      "x264_crf1_aq0",      "-c:v libx264 -preset slow -crf 1 -pix_fmt yuv420p -x264-params aq-mode=0:deblock=0,0:psy-rd=0,0 -tune grain";
      "x264_qp1",           "-c:v libx264 -preset slow -qp 1 -pix_fmt yuv420p -x264-params aq-mode=0:deblock=0,0" };
rows = {};
for k = 1:size(V, 1)
    f = fullfile(out, V{k, 1} + ".mp4");
    cmd = sprintf('"%s" -y -hide_banner -loglevel error -i "%s" -t 30 %s -an "%s"', ff, vp, V{k, 2}, f);
    [st, msg] = system(cmd);
    if st ~= 0, fprintf('%s: ffmpeg failed: %s\n', V{k, 1}, msg); continue; end
    kb = dir(f).bytes * 8 / 1000 / 30;
    try
        r = VideoReader(f); readable = r.NumFrames > 0; nF = r.NumFrames; clear r
        R = runOffline(f, Reference=gt);
        rows(end+1, :) = {V{k, 1}, V{k, 2}, kb, readable, nF, R.metrics.MAE, R.metrics.coverage}; %#ok<AGROW>
        fprintf('%-18s %7.0f kbps  frames %d  MAE %.2f  coverage %.0f%%\n', V{k, 1}, kb, nF, R.metrics.MAE, R.metrics.coverage);
    catch e
        rows(end+1, :) = {V{k, 1}, V{k, 2}, kb, false, 0, NaN, NaN}; %#ok<AGROW>
        fprintf('%-18s %7.0f kbps  NOT READABLE / failed: %s\n', V{k, 1}, kb, e.message);
    end
end
T = cell2table(rows, 'VariableNames', {'Variant', 'Args', 'Kbps', 'Readable', 'Frames', 'MAE', 'Coverage'});
disp(T);
end
