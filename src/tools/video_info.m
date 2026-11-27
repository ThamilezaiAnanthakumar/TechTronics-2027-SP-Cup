function video_info(videoPath, opts)
%VIDEO_INFO  Print header facts and timestamp checks of a video.
%   video_info(videoPath)
%   video_info(videoPath, GtPath="...ground_truth.txt", N=5)

arguments
    videoPath (1,1) string {mustBeFile}
    opts.GtPath string = string.empty
    opts.N (1,1) double {mustBePositive, mustBeInteger} = 5
end

hdr = read_video_header(videoPath);
fprintf("\n=== %s\n", hdr.Path);
fprintf("resolution      : %d x %d\n", hdr.Width, hdr.Height);
fprintf("frame rate      : %.3f fps\n", hdr.FrameRate);
fprintf("pixel format    : %s\n", hdr.VideoFormat);
fprintf("frames (header) : %d\n", hdr.NumFrames);
fprintf("duration        : %.3f s\n", hdr.Duration);

t = read_frame_times(videoPath);
n = min(opts.N, numel(t));
k = (1:n).';
tAssumed = (k - 1) / hdr.FrameRate;
fprintf("first %d timestamps (video | (k-1)/fps):\n", n);
fprintf("   %8.4f | %8.4f\n", [t(k) tAssumed].');
fprintf("max |difference| : %.3f ms\n", max(abs(t(k) - tAssumed)) * 1000);

dt = diff(t);
fprintf("decoded frames  : %d (%s)\n", numel(t), ok_str(numel(t) == hdr.NumFrames));
fprintf("real duration   : %.3f s\n", t(end) - t(1));
if ~isempty(dt)
    fprintf("mean fps        : %.3f\n", 1 / mean(dt));
    fprintf("frame interval  : min %.3f ms, max %.3f ms\n", min(dt) * 1000, max(dt) * 1000);
end

if ~isempty(opts.GtPath)
    gt = load_ubfc_gt(opts.GtPath);
    fprintf("GT samples      : %d (%s)\n", numel(gt.t), ok_str(numel(gt.t) == numel(t)));
    fprintf("GT duration     : %.3f s\n", gt.t(end) - gt.t(1));
    fprintf("GT mean rate    : %.3f Hz\n", 1 / mean(diff(gt.t)));
end
fprintf("\n");
end

function s = ok_str(tf)
if tf, s = "OK"; else, s = "MISMATCH"; end
end
