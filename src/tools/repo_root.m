function root = repo_root()
%REPO_ROOT  Absolute path of the repository root.
root = string(fileparts(fileparts(fileparts(mfilename("fullpath")))));
end
