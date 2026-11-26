function make_aipex_split()
%MAKE_AIPEX_SPLIT Write the frozen AiPEX dev/test split (README Section 8.5).
%   30 % of subjects held out as AiPEX-test, stratified by country, rng(2027).
%   One video id = one subject (dataset README: three ROI crops are generated
%   from each subject's single facial video). Writes shared/splits/aipex_test.txt
%   and aipex_dev.txt, one "<Country>/<videoId>" per line. Run once; never change.
repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repoRoot, 'config'));
cfg = getConfig();
outDir = fullfile(repoRoot, 'shared', 'splits');
if isfile(fullfile(outDir, 'aipex_test.txt'))
    error('make_aipex_split:exists', 'Split already exists; it is frozen.');
end
rng(2027, 'twister');
test = strings(0, 1); dev = strings(0, 1);
for c = string(cfg.aipexCountries)
    gt = readtable(fullfile(cfg.aipexRoot, 'Ground_Truth', c + "_GT.csv"), ...
        'TextType', 'string', 'VariableNamingRule', 'preserve');
    ids = strtrim(gt{:, 1});
    assert(numel(unique(ids)) == numel(ids), 'duplicate ids in %s', c);
    nTest = round(0.3 * numel(ids));
    perm = randperm(numel(ids));
    test = [test; c + "/" + sort(ids(perm(1:nTest)))];       %#ok<AGROW>
    dev  = [dev;  c + "/" + sort(ids(perm(nTest+1:end)))];   %#ok<AGROW>
    fprintf('%s: %d videos -> %d test, %d dev\n', c, numel(ids), nTest, numel(ids) - nTest);
end
if ~isfolder(outDir), mkdir(outDir); end
writelines(test, fullfile(outDir, 'aipex_test.txt'));
writelines(dev,  fullfile(outDir, 'aipex_dev.txt'));
fprintf('wrote %d test / %d dev ids to %s\n', numel(test), numel(dev), outDir);
end
