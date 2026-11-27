function t = read_frame_times(videoPath, maxFrames)
%READ_FRAME_TIMES  Timestamp (s) of every decoded frame, read from the video.
%   t = read_frame_times(videoPath)
%   t = read_frame_times(videoPath, maxFrames)
%   t  Nx1 double, N = frames actually decoded (never computed as k/fps)

arguments
    videoPath (1,1) string {mustBeFile}
    maxFrames (1,1) double {mustBePositive} = Inf
end

v = VideoReader(videoPath);
t = zeros(min(maxFrames, v.NumFrames), 1);
k = 0;
while hasFrame(v) && k < maxFrames
    k = k + 1;
    t(k) = v.CurrentTime;        % time of the frame about to be read
    readFrame(v);
end
t = t(1:k);
end
