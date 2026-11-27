function info = lowBandwidth(videoPath, opts)
%LOWBANDWIDTH C5 low-bandwidth simulation (README 1.4 / 5): frame decimation + MPEG-4 re-encode.
%   info = lowBandwidth(videoPath)                       10 fps, <= 500 kbps
%   info = lowBandwidth(v, Fps=6, BitrateKbps=250, MaxFrames=300)
%   Decimation uses the real source timestamps (VideoReader CurrentTime): for each target time
%   t0 + k/Fps the source frame with the nearest timestamp is kept (no frame blending).
%   VideoWriter 'MPEG-4' exposes Quality, not a bitrate, so the clip is encoded with Quality
%   = QualitySteps(1), QualitySteps(2), ... until the MEASURED bitrate (8*bytes/duration) is
%   <= BitrateKbps; the highest quality that meets the target is kept. All attempts are logged
%   in the sidecar. If no step meets the target, target_met = false: re-encode with ffmpeg
%   (-b:v 500k -maxrate 500k -bufsize 1000k), which is not installed here (pending).
%   Output: <OutRoot>/LowBandwidth_C5_<fps>fps_<kbps>k/<subject>.mp4 + .json sidecar.
arguments
    videoPath (1,1) string {mustBeFile}
    opts.Fps (1,1) double {mustBePositive} = 10
    opts.BitrateKbps (1,1) double {mustBePositive} = 500
    opts.QualitySteps (1,:) double = [75 60 50 40 30 20 15 10 5 2 0]
    opts.Condition (1,1) string = ""
    opts.OutRoot (1,1) string = ""
    opts.Subject (1,1) string = ""
    opts.MaxFrames (1,1) double = Inf
end
if opts.OutRoot == "", opts.OutRoot = degradeDefaultRoot(); end
cond = opts.Condition;
if cond == "", cond = sprintf("LowBandwidth_C5_%gfps_%gk", opts.Fps, opts.BitrateKbps); end
v = VideoReader(videoPath);
src = degradeSource(v, videoPath);
subject = degradeSubject(videoPath, opts.Subject);
if opts.Fps > v.FrameRate
    error('lowBandwidth:fps', 'Target %g fps exceeds the source frame rate %g fps.', opts.Fps, v.FrameRate);
end

% 1) decimate by real timestamps into a lossless intermediate (keeps memory flat)
[wi, tmpFile] = degradeWriter(opts.OutRoot, cond, [subject '_decimated_tmp'], "Archival", opts.Fps);
open(wi);
n = min(opts.MaxFrames, v.NumFrames); k = 0; tSel = []; idxSel = [];
prevI = []; prevT = NaN; nextT = NaN;
while hasFrame(v) && k < n
    k = k + 1; t = v.CurrentTime; I = readFrame(v);
    if k == 1, nextT = t; end
    while t >= nextT                                  % current frame passed the target time
        if ~isempty(prevI) && abs(prevT - nextT) < abs(t - nextT) && (isempty(idxSel) || idxSel(end) ~= k - 1)
            writeVideo(wi, prevI); tSel(end+1) = prevT; idxSel(end+1) = k - 1; %#ok<AGROW>
        elseif isempty(idxSel) || idxSel(end) ~= k
            writeVideo(wi, I); tSel(end+1) = t; idxSel(end+1) = k; %#ok<AGROW>
        end
        nextT = nextT + 1 / opts.Fps;
    end
    prevI = I; prevT = t;
end
close(wi); clear v
nSrc = k;

% 2) MPEG-4 encode, lowering Quality until the measured bitrate meets the target
outDir = fullfile(opts.OutRoot, cond);
outFile = char(fullfile(outDir, [subject '.mp4']));
attempts = struct('quality', {}, 'bitrate_kbps', {}, 'bytes', {});
for q = opts.QualitySteps
    encodeMp4(tmpFile, outFile, opts.Fps, q);
    d = dir(outFile); r = VideoReader(outFile); dur = r.Duration; clear r
    attempts(end+1) = struct('quality', q, 'bitrate_kbps', 8 * d.bytes / dur / 1000, 'bytes', d.bytes); %#ok<AGROW>
    if attempts(end).bitrate_kbps <= opts.BitrateKbps, break; end
end
delete(tmpFile);
met = attempts(end).bitrate_kbps <= opts.BitrateKbps;

% 3) measured fps from the output's own timestamps
r = VideoReader(outFile); tOut = [];
while hasFrame(r), tOut(end+1) = r.CurrentTime; readFrame(r); end %#ok<AGROW>
clear r
p = struct('generator', 'lowBandwidth', 'constraint', 'C5', 'target_fps', opts.Fps, ...
    'target_bitrate_kbps', opts.BitrateKbps, 'encoder', 'VideoWriter MPEG-4 (H.264, Media Foundation)', ...
    'quality', attempts(end).quality, 'quality_attempts', attempts, 'target_met', met, ...
    'decimation', 'nearest real source timestamp to t0 + k/fps', ...
    'source_frames_read', nSrc, 'source_frame_index', idxSel, 'source_t_s', tSel, ...
    'ffmpeg_reencode', 'pending: ffmpeg not installed; bitrate controlled via VideoWriter Quality only');
info = degradeSidecar(outFile, cond, subject, src, p, "MPEG-4");
info.fps_measured = 1 / median(diff(tOut));
info.target_met = met;
degradeAppend(info, 'fps_measured', info.fps_measured);
end

function encodeMp4(inFile, outFile, fps, q)
r = VideoReader(inFile);
w = VideoWriter(outFile, 'MPEG-4'); w.FrameRate = fps; w.Quality = q;
open(w);
while hasFrame(r), writeVideo(w, readFrame(r)); end
close(w);
end
