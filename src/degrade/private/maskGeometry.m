function [maskPoly, lowerPoly] = maskGeometry(L, box, topFrac)
%MASKGEOMETRY C7 mask polygon and the lower-face reference polygon (private).
%   Landmark mode (L = 478x2 MediaPipe, 1-based pixels): the face-oval contour is cut by a
%   line perpendicular to the face axis (forehead 10 -> chin 152) through a point topFrac of
%   the way from the subnasale (landmark 2) to the chin; the part below is the mask. The
%   lower face = the oval below the nose tip (landmark 1), same construction.
%   Box mode (box = [x y w h], synthetic tests): nose tip at 0.60 h, subnasale at 0.65 h,
%   chin at h; mask = hexagon from the subnasale line to the chin; lower face = box rows 0.60-1.
%   Polygons: Kx2 [x y].
if ~isempty(L)
    oval = [234 93 132 58 172 136 150 149 176 148 152 377 400 378 379 365 397 288 361 323 454] + 1;
    P = L(oval, :);
    u = L(153, :) - L(11, :); u = u / norm(u);
    top = L(3, :) + topFrac * dot(L(153, :) - L(3, :), u) * u;
    maskPoly = cutBelow(P, top, u);
    lowerPoly = cutBelow(P, L(2, :), u);
else
    x = box(1); y = box(2); w = box(3); h = box(4); ys = 0.65 + topFrac * 0.35;
    f = [0.08 ys; 0.92 ys; 0.88 0.85; 0.66 1.0; 0.34 1.0; 0.12 0.85];
    maskPoly = [x + f(:, 1) * w, y + f(:, 2) * h];
    lowerPoly = [x y + 0.60 * h; x + w y + 0.60 * h; x + w y + h; x y + h];
end
end

function Q = cutBelow(P, p0, u)
% Part of the open contour P (ordered side -> chin -> side) on the chin side of the line
% through p0 perpendicular to u, closed along that line.
s = (P - p0) * u';
Q = zeros(0, 2);
for i = 1:size(P, 1)
    if i > 1 && sign(s(i)) ~= sign(s(i-1)) && s(i) ~= s(i-1)
        a = s(i-1) / (s(i-1) - s(i)); Q(end+1, :) = P(i-1, :) + a * (P(i, :) - P(i-1, :)); %#ok<AGROW>
    end
    if s(i) >= 0, Q(end+1, :) = P(i, :); end %#ok<AGROW>
end
end
