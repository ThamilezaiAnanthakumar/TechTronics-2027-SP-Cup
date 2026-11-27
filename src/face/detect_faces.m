function [bboxes, scores] = detect_faces(fd, I, p, roi)
%DETECT_FACES  Detect all faces in a frame with RetinaFace.
%   [bboxes, scores] = detect_faces(fd, I, p)        whole frame
%   [bboxes, scores] = detect_faces(fd, I, p, roi)   inside roi = [x y w h]
%
%   fd      faceDetector object (create once: faceDetector(p.det.model))
%   I       HxWx3 uint8 RGB frame
%   p       parameters from face_params
%   bboxes  Mx4 [x y w h] in frame coordinates, scores Mx1

arguments
    fd
    I (:,:,3) uint8
    p struct
    roi double = []
end

[H, W, ~] = size(I);
s = p.det.upscale;
if s ~= 1
    J = imresize(I, s, "bilinear");
else
    J = I;
end
args = {"Threshold", p.det.threshold, "MinSize", max(1, round(p.det.minSize * s)), ...
        "ExecutionEnvironment", p.det.device};

if isempty(roi)
    [b, sc] = detect(fd, J, args{:});
else
    r = [(roi(1) - 0.5) * s + 0.5, (roi(2) - 0.5) * s + 0.5, roi(3) * s, roi(4) * s];
    r = clip_box(r, [size(J, 1) size(J, 2)], true);
    if any(r(3:4) < 16)
        bboxes = zeros(0, 4);  scores = zeros(0, 1);
        return
    end
    [b, sc] = detect(fd, J, r, args{:});
end

b = double(b);
scores = double(sc(:));
if isempty(b)
    bboxes = zeros(0, 4);  scores = zeros(0, 1);
    return
end
if s ~= 1
    b = [(b(:,1) - 0.5) / s + 0.5, (b(:,2) - 0.5) / s + 0.5, b(:,3) / s, b(:,4) / s];
end
bboxes = clip_box(b, [H W]);
keep = bboxes(:,3) >= 2 & bboxes(:,4) >= 2;
bboxes = bboxes(keep, :);
scores = scores(keep);
end
