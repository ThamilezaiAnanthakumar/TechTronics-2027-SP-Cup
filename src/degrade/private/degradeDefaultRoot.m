function r = degradeDefaultRoot()
%DEGRADEDEFAULTROOT <dataset parent>\work\degraded (D:\SPCup Data\work\degraded on the team PC).
%   Derived from getConfig().datasetRoot so no new machine-specific path is introduced.
cfg = getConfig();
r = fullfile(fileparts(cfg.datasetRoot), 'work', 'degraded');
end
