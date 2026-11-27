function boxes = roiBoxes(faceBox, cfg, imSize)
%ROIBOXES ROI rectangles relative to a face box.
%   boxes = roiBoxes(faceBox, cfg, imSize)
%   faceBox : [x y w h] (pixels)
%   imSize  : [rows cols], used to clip
%   boxes.(name) = [r1 r2 c1 c2] integer pixel ranges for each base ROI
%   (forehead, leftCheek, rightCheek). 'combined' is handled by the caller.

names = fieldnames(cfg.roi);
x = faceBox(1); y = faceBox(2); w = faceBox(3); h = faceBox(4);
for k = 1:numel(names)
    r = cfg.roi.(names{k});           % [x1 x2 y1 y2] fractions
    c1 = round(x + r(1)*w); c2 = round(x + r(2)*w);
    r1 = round(y + r(3)*h); r2 = round(y + r(4)*h);
    c1 = min(max(c1, 1), imSize(2)); c2 = min(max(c2, 1), imSize(2));
    r1 = min(max(r1, 1), imSize(1)); r2 = min(max(r2, 1), imSize(1));
    boxes.(names{k}) = [r1 r2 c1 c2];
end
end
