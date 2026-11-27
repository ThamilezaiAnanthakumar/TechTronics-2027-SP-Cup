function h = gitInfo()
%GITINFO Short git hash of the repository (with "-dirty" if there are uncommitted changes), or 'unknown'.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
[s, h] = system(sprintf('git -C "%s" rev-parse --short HEAD', root));
if s ~= 0, h = 'unknown'; return; end
h = strtrim(h);
[s, d] = system(sprintf('git -C "%s" status --porcelain --untracked-files=no', root));
if s == 0 && ~isempty(strtrim(d)), h = [h '-dirty']; end
end
