function cache_ubfc_test_traces()
%CACHE_UBFC_TEST_TRACES Extract (cache only) anatomical-patch traces for the UBFC2 test subjects.
%   Front-end only: no HR, no metrics. Used after the parameter freeze (README 8.2).
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'config')); addpath(genpath(fullfile(root, 'src')));
cfg = getConfig();
dev = "subject" + [1 3 5 8 9 10 11 12 13 14 15 16];
subs = setdiff(string(listUBFCSubjects(cfg)), dev, 'stable');
fprintf('UBFC test subjects: %d\n', numel(subs));
for s = subs
    c = cfg; c.subject = char(s); [vp, ~] = loadUBFC(c);
    out = fullfile("D:\SPCup Data\work\anat_traces", s + ".mat");
    if isfile(out), continue; end
    t0 = tic; anatomicalPatchTraces(vp, CacheFile=out); fprintf('%s cached (%.0f s)\n', s, toc(t0));
end
fprintf('== UBFC TEST CACHE DONE\n');
end
