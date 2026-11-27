function J = draw_mask(I, box, p)
%DRAW_MASK  Paint a synthetic surgical mask over the lower face (C7 test data).
%   J = draw_mask(I, box, p)
%   box: 1x4 target face box; polygon p.clip.maskPoly (face-box fractions),
%   colour p.clip.maskColor.

if isempty(box) || any(isnan(box))
    J = I;
    return
end
[H, W, ~] = size(I);
x = box(1) + p.clip.maskPoly(:, 1) * box(3);
y = box(2) + p.clip.maskPoly(:, 2) * box(4);
a = imgaussfilt(double(poly2mask(x, y, H, W)), 1.0);
shade = 1 - 0.15 * min(max(((1:H).' - min(y)) / max(max(y) - min(y), 1), 0), 1);
colour = reshape(double(p.clip.maskColor), 1, 1, 3) .* shade;
J = uint8(a .* colour + (1 - a) .* double(I));
end
