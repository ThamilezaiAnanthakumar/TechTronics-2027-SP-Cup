function info = degradeRun(videoPath, frameFcn, condition, params, opts)
%DEGRADERUN Shared frame loop of the constraint generators (private).
%   info = degradeRun(videoPath, frameFcn, condition, params, opts)
%   frameFcn(I, k, t) -> uint8 frame. Writes <opts.OutRoot>/<condition>/<subject>.<ext>
%   with a lossless writer by default (MJ2 'Archival'), so the only degradation in the
%   file is the simulated one, then writes the JSON sidecar (degradeSidecar).
%   opts: OutRoot, Subject, MaxFrames, Profile, Overwrite.
v = VideoReader(videoPath);
src = degradeSource(v, videoPath);
subject = degradeSubject(videoPath, opts.Subject);
[w, outFile] = degradeWriter(opts.OutRoot, condition, subject, opts.Profile, v.FrameRate);
open(w); c = onCleanup(@() close(w));
n = min(opts.MaxFrames, v.NumFrames); k = 0; tSrc = zeros(n, 1);
while hasFrame(v) && k < n
    k = k + 1;
    tSrc(k) = v.CurrentTime;
    writeVideo(w, frameFcn(readFrame(v), k, tSrc(k)));
end
clear c
params.source_frames_used = k;
params.source_t_range_s = [tSrc(1) tSrc(k)];
info = degradeSidecar(outFile, condition, subject, src, params, opts.Profile);
end
