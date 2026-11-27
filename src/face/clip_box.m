function c = clip_box(b, frameSize, asInteger)
%CLIP_BOX  Clip [x y w h] boxes (Mx4) to an image of size [H W].
%   c = clip_box(b, frameSize, true) returns integer boxes usable as a ROI
%   for detect / detectMinEigenFeatures (last pixel x+w-1 <= W).

if nargin < 3, asInteger = false; end
H = frameSize(1);  W = frameSize(2);
x1 = b(:,1);  y1 = b(:,2);
x2 = b(:,1) + b(:,3);  y2 = b(:,2) + b(:,4);
if asInteger
    x1 = round(x1);  y1 = round(y1);  x2 = round(x2);  y2 = round(y2);
end
x1 = max(1, x1);      y1 = max(1, y1);
x2 = min(W + 1, x2);  y2 = min(H + 1, y2);
c = [x1, y1, max(0, x2 - x1), max(0, y2 - y1)];
end
