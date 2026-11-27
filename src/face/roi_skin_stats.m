function [rgbSum, nSkin, nPoly, M, rect] = roi_skin_stats(I, poly, p)
%ROI_SKIN_STATS  Skin pixels inside one ROI polygon.
%   [rgbSum, nSkin, nPoly, M, rect] = roi_skin_stats(I, poly, p)
%
%   I       HxWx3 uint8 frame,  poly  Kx2 [x y] polygon
%   rgbSum  1x3 sum of R,G,B over skin pixels (mean = rgbSum / nSkin)
%   nSkin   skin pixels inside the polygon,  nPoly  all pixels inside it
%   M, rect skin mask of the bounding rectangle rect = [x y w h]

[H, W, ~] = size(I);
rgbSum = zeros(1, 3);  nSkin = 0;  nPoly = 0;  M = false(0);  rect = zeros(1, 4);
if isempty(poly) || any(~isfinite(poly(:))), return; end

x1 = max(1, floor(min(poly(:,1))));  x2 = min(W, ceil(max(poly(:,1))));
y1 = max(1, floor(min(poly(:,2))));  y2 = min(H, ceil(max(poly(:,2))));
if x2 <= x1 || y2 <= y1, return; end
rect = [x1, y1, x2 - x1 + 1, y2 - y1 + 1];

inPoly = poly2mask(poly(:,1) - x1 + 1, poly(:,2) - y1 + 1, rect(4), rect(3));
Pch = I(y1:y2, x1:x2, :);
M = inPoly & skin_mask(Pch, p);
nPoly = nnz(inPoly);
nSkin = nnz(M);
if nSkin > 0
    X = reshape(double(Pch), [], 3);
    rgbSum = sum(X(M(:), :), 1);
end
end
