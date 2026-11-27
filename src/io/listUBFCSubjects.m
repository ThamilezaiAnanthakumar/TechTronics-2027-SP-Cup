function subjects = listUBFCSubjects(cfg)
%LISTUBFCSUBJECTS Names of UBFC subjects that have a vid.avi under cfg.dataRoot.
%   subjects = listUBFCSubjects(cfg)  -> cellstr sorted by subject number,
%   e.g. {'subject1','subject3','subject4'}. Looks directly under dataRoot
%   and one wrapper level down (Google Drive export folders).

f = [dir(fullfile(cfg.dataRoot, 'subject*', 'vid.avi')); ...
     dir(fullfile(cfg.dataRoot, '*', 'subject*', 'vid.avi'))];
names = cell(numel(f), 1);
for k = 1:numel(f)
    [~, names{k}] = fileparts(f(k).folder);
end
names = unique(names);
num = cellfun(@(s) str2double(regexp(s, '\d+', 'match', 'once')), names);
[~, i] = sort(num);
subjects = names(i)';
end
