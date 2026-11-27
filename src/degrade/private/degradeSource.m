function src = degradeSource(v, videoPath)
%DEGRADESOURCE Properties of the source video for the sidecar (private).
d = dir(videoPath);
src = struct('path', char(videoPath), 'width', v.Width, 'height', v.Height, ...
    'fps', v.FrameRate, 'duration_s', v.Duration, 'num_frames', v.NumFrames, ...
    'bytes', d.bytes, 'bitrate_kbps', 8 * d.bytes / max(v.Duration, eps) / 1000);
end
