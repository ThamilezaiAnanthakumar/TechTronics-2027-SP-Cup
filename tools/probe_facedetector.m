%PROBE_FACEDETECTOR Check RetinaFace faceDetector availability and its outputs.
%   Records: MATLAB version, model names that load, detect() output arity and
%   whether landmarks are returned. Uses a synthetic test image only.
fprintf('MATLAB %s\n', version);
for m = ["small-network" "large-network"]
    try
        fd = faceDetector(m); fprintf('faceDetector("%s"): OK (%s)\n', m, class(fd));
    catch e
        fprintf('faceDetector("%s"): FAIL %s\n', m, e.message);
    end
end
fd = faceDetector("small-network");
disp(properties(fd)');
disp(methods(fd)');
I = imread('visionteam1.jpg');           % shipped with Computer Vision Toolbox
[b, s] = detect(fd, I);
fprintf('detect: %d faces, bbox %s, scores %s\n', size(b,1), class(b), class(s));
try
    [b, s, lm] = detect(fd, I);
    fprintf('3rd output: class %s size %s\n', class(lm), mat2str(size(lm)));
catch e
    fprintf('3rd output: not supported (%s)\n', e.message);
end
h = help('faceDetector/detect'); disp(h(1:min(end, 1500)));
