function M = skin_mask(P, p)
%SKIN_MASK  Skin pixels of an RGB patch (hxwx3 uint8) -> hxw logical.
%   YCbCr chroma box p.skin.cb / p.skin.cr, rejecting dark (Y < p.skin.yMin)
%   and saturated (any channel >= p.skin.satMax) pixels.

Y = rgb2ycbcr(P);
M = Y(:,:,2) >= p.skin.cb(1) & Y(:,:,2) <= p.skin.cb(2) & ...
    Y(:,:,3) >= p.skin.cr(1) & Y(:,:,3) <= p.skin.cr(2) & ...
    Y(:,:,1) >= p.skin.yMin  & max(P, [], 3) < p.skin.satMax;
end
