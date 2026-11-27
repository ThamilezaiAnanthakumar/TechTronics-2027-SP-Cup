function V = loadAiPEX(cfg)
%LOADAIPEX Index the AiPEX rppg_biases dataset (pre-cropped ROI videos).
%   V = loadAiPEX(cfg) returns a table, one row per subject video:
%     Country, Video ('video12'), gtHR (mean HR from the Masimo oximeter, bpm),
%     and one path column per ROI in cfg.aipexRoiFiles (forehead, leftCheek,
%     rightCheek). Errors if a GT entry has a missing video file.
%   Layout: Data/<Country>/videoN_<roi>.avi, Ground_Truth/<Country>_GT.csv

rois = fieldnames(cfg.aipexRoiFiles);
rows = {};
for c = 1:numel(cfg.aipexCountries)
    country = cfg.aipexCountries{c};
    gtFile = fullfile(cfg.aipexRoot, 'Ground_Truth', [country '_GT.csv']);
    if ~isfile(gtFile)
        error('loadAiPEX:noGT', 'Ground truth not found: %s', gtFile);
    end
    % Two columns: video name, mean HR. Headers have stray spaces / BOM.
    G = readtable(gtFile, 'ReadVariableNames', false, 'NumHeaderLines', 1, ...
                  'Delimiter', ',', 'TextType', 'string');
    for k = 1:height(G)
        vid = strtrim(G{k, 1});
        if strlength(vid) == 0, continue; end
        paths = cell(1, numel(rois));
        for r = 1:numel(rois)
            p = fullfile(cfg.aipexRoot, 'Data', country, ...
                sprintf('%s_%s.avi', vid, cfg.aipexRoiFiles.(rois{r})));
            if ~isfile(p)
                error('loadAiPEX:noVideo', 'Missing video: %s', p);
            end
            paths{r} = p;
        end
        rows(end+1, :) = [{string(country), vid, double(G{k, 2})}, paths]; %#ok<AGROW>
    end
end
V = cell2table(rows, 'VariableNames', [{'Country', 'Video', 'gtHR'}, rois']);
end
