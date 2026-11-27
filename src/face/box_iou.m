function iou = box_iou(a, B)
%BOX_IOU  Intersection over union of box a (1x4) with boxes B (Mx4), [x y w h].
%   Returns Mx1; boxes containing NaN give 0.

if isempty(B)
    iou = zeros(0, 1);
    return
end
x1 = max(a(1), B(:,1));
y1 = max(a(2), B(:,2));
x2 = min(a(1) + a(3), B(:,1) + B(:,3));
y2 = min(a(2) + a(4), B(:,2) + B(:,4));
inter = max(0, x2 - x1) .* max(0, y2 - y1);
iou = inter ./ (a(3) * a(4) + B(:,3) .* B(:,4) - inter);
iou(~isfinite(iou)) = 0;
end
