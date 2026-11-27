function [s, n] = roiSumRGB(frame, box, cfg)
%ROISUMRGB Sum of R,G,B over the (skin) pixels of one rectangular ROI.
%   [s, n] = roiSumRGB(frame, box, cfg)
%   frame : H x W x 3 image (any numeric class)
%   box   : [r1 r2 c1 c2]
%   s     : 1 x 3 channel sums (double); n : number of pixels used
%   Sums (not means) are returned so ROIs can be pooled pixel-weighted.

patch = frame(box(1):box(2), box(3):box(4), :);
px = reshape(double(patch), [], 3);
if isempty(px)
    s = nan(1, 3); n = 0;
    return
end

if cfg.useSkinMask
    ycc = rgb2ycbcr(im2uint8(patch));
    cb = double(ycc(:,:,2)); cr = double(ycc(:,:,3));
    mask = cb >= cfg.skinCb(1) & cb <= cfg.skinCb(2) & ...
           cr >= cfg.skinCr(1) & cr <= cfg.skinCr(2);
    if nnz(mask) >= cfg.minSkinPixels
        px = px(mask(:), :);
    end
end
s = sum(px, 1);
n = size(px, 1);
end
