function R = roi_polygons(anchorBox, A, p)
%ROI_POLYGONS  ROI polygons of the target face in the current frame.
%   R = roi_polygons(anchorBox, A, p)
%
%   anchorBox  1x4 target box at the last anchor
%   A          3x3 transform anchor frame -> current frame (KLT)
%   R          struct: one 4x2 [x y] polygon per region in p.roi.names,
%              plus R.box, the 1x4 bounding box of the moved face box

x = anchorBox(1);  y = anchorBox(2);  w = anchorBox(3);  h = anchorBox(4);
for name = p.roi.names
    r = p.roi.(name);
    P = [x + r(1)*w, y + r(2)*h;
         x + r(3)*w, y + r(2)*h;
         x + r(3)*w, y + r(4)*h;
         x + r(1)*w, y + r(4)*h];
    R.(name) = apply_tform(A, P);
end
C = apply_tform(A, [x y; x+w y; x+w y+h; x y+h]);
R.box = [min(C(:,1)), min(C(:,2)), max(C(:,1)) - min(C(:,1)), max(C(:,2)) - min(C(:,2))];
end
