function exportParityData(subjects)
%EXPORTPARITYDATA Export cached RGB traces + MATLAB HR results to JSON so the
%   JavaScript port (deploy/web/rppg.js) can be checked against MATLAB.
%   exportParityData()                      subjects with outputs/<s>/traces.mat
%   exportParityData({'subject1'})
%   Writes outputs/web_parity/<subject>.json (gitignored).

webDir   = fileparts(fileparts(mfilename('fullpath')));
repoRoot = fileparts(fileparts(webDir));
addpath(fullfile(repoRoot, 'config'));
addpath(genpath(fullfile(repoRoot, 'src')));
cfg = getConfig();

if nargin < 1
    d = dir(fullfile(cfg.resultsRoot, '*', 'traces.mat'));
    subjects = unique({d.folder});
    subjects = cellfun(@(f) regexprep(f, '.*[\\/]', ''), subjects, 'UniformOutput', false);
end
outDir = fullfile(cfg.resultsRoot, 'web_parity');
if ~isfolder(outDir), mkdir(outDir); end

for s = 1:numel(subjects)
    S = load(fullfile(cfg.resultsRoot, subjects{s}, 'traces.mat'), 'traces');
    tr = S.traces;
    J = struct('subject', subjects{s}, 'fps', tr.fps, 'cfg', cfgSubset(cfg));
    for r = 1:numel(cfg.roiSet)
        roi = cfg.roiSet{r};
        J.rgb.(roi) = tr.rgb.(roi);
        for m = 1:numel(cfg.methods)
            method = cfg.methods{m};
            p  = rgbToPulse(tr.rgb.(roi), tr.fps, method, cfg);
            pf = preprocessPulse(p, tr.fps, cfg);
            [tHR, hr] = estimateHR(pf, tr.fps, cfg);
            J.hr.(method).(roi) = hr;
            J.pf.(method).(roi) = pf;
        end
    end
    J.tHR = tHR;
    resFile = fullfile(cfg.resultsRoot, subjects{s}, 'hr_results.mat');
    if isfile(resFile)
        Rs = load(resFile, 'hrRef');
        if numel(Rs.hrRef) == numel(tHR), J.hrRef = Rs.hrRef; end
    end
    f = fullfile(outDir, [subjects{s} '.json']);
    fid = fopen(f, 'w'); fwrite(fid, jsonencode(J)); fclose(fid);
    fprintf('Wrote %s (%d frames, %.3f fps)\n', f, size(tr.rgb.combined, 1), tr.fps);
end
end

function c = cfgSubset(cfg)
c = struct('hrBand', cfg.hrBand, 'filterOrder', cfg.filterOrder, ...
    'detrendLambda', cfg.detrendLambda, 'posWindowSec', cfg.posWindowSec, ...
    'hrWindowSec', cfg.hrWindowSec, 'hrStepSec', cfg.hrStepSec, 'nfftMin', cfg.nfftMin);
end
