function gt = load_ubfc_gt(gtPath)
%LOAD_UBFC_GT  Load a UBFC-rPPG DATASET_2 ground_truth.txt.
%   gt = load_ubfc_gt(gtPath)
%   gt fields (Kx1 double): ppg (contact PPG), hr (sensor HR, bpm), t (s)
%   Reference HR for evaluation should be computed from gt.ppg.

arguments
    gtPath (1,1) string {mustBeFile}
end

lines = strtrim(splitlines(string(fileread(gtPath))));
lines = lines(lines ~= "");
if numel(lines) ~= 3
    error("load_ubfc_gt:format", "%s: expected 3 lines, found %d.", gtPath, numel(lines));
end
rows = cell(3, 1);
for i = 1:3
    rows{i} = sscanf(char(lines(i)), "%f");
end
lens = cellfun(@numel, rows);
if any(lens ~= lens(1)) || lens(1) == 0
    error("load_ubfc_gt:format", "%s: line lengths differ (%s).", gtPath, mat2str(lens.'));
end
gt.ppg = rows{1};
gt.hr  = rows{2};
gt.t   = rows{3};
end
