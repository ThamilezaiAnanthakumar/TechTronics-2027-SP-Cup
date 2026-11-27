function s = degradeSubject(videoPath, s)
%DEGRADESUBJECT Output base name: explicit, else the parent folder (UBFC 'subjectN'), else the file name.
if strlength(s) > 0, s = char(s); return; end
[d, f] = fileparts(char(videoPath));
[~, parent] = fileparts(d);
if startsWith(parent, 'subject', 'IgnoreCase', true), s = parent; else, s = f; end
end
