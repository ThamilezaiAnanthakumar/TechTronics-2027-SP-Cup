function cfg = getConfig(varargin)
%GETCONFIG Single source of truth for all pipeline parameters.
%   cfg = getConfig()                 returns the defaults.
%   cfg = getConfig('name',value,...) overrides top-level fields, e.g.
%         getConfig('maxFrames',900,'subject','subject3')
%
%   No other file should contain magic numbers or hardcoded paths.

repoRoot = fileparts(fileparts(mfilename('fullpath')));
cfg.repoRoot = repoRoot;

% ---------------- Paths ----------------
% Base folder holding all datasets (UBFC-rPPG, AiPEX, ...). Machine-specific:
% change this one line on another computer.
cfg.datasetRoot = 'D:\SPCup Data\dataset';
% UBFC DATASET_2 root. The subject folder is searched for directly under
% this folder or one level down (handles Google Drive export wrappers such
% as "subject1-2026...-001/subject1").
cfg.dataRoot    = fullfile(cfg.datasetRoot, 'UBFC-rPPG', 'Dataset2');
cfg.subject     = 'subject1';
cfg.videoPath   = '';   % optional explicit video file; '' = <subject folder>/vid.avi
cfg.resultsRoot = fullfile(repoRoot, 'outputs');   % gitignored; never write into data
cfg.privateSubjects = [21 27];  % UBFC: never save face images for these

% AiPEX rppg_biases (Dasari et al. 2021): pre-cropped ROI videos + mean GT HR
cfg.aipexRoot      = fullfile(cfg.datasetRoot, 'AiPEX rppg_biases', 'rppg_biases-main');
cfg.aipexCountries = {'India', 'Sierra_Leone'};
% file suffix -> pipeline ROI name
cfg.aipexRoiFiles  = struct('forehead', 'forehead', 'leftCheek', 'leftcheek', 'rightCheek', 'rightcheek');
cfg.useParallel    = true;  % parfor over videos in batch scripts

% ---------------- Face detection / tracking ----------------
cfg.detectEvery   = 30;     % re-run face detection every N frames, KLT in between
cfg.maxFrames     = Inf;    % e.g. 900 for quick tests (30 s at 30 fps)
cfg.minTrackPts   = 10;     % fewer valid KLT points -> tracking considered lost
cfg.progressEvery = 300;    % print progress every N frames

% ROIs relative to the face box: [x1 x2 y1 y2] as fractions of w / h
cfg.roi.forehead   = [0.30 0.70 0.08 0.25];
cfg.roi.leftCheek  = [0.15 0.38 0.50 0.72];
cfg.roi.rightCheek = [0.62 0.85 0.50 0.72];

% Skin mask (YCbCr) inside each ROI
cfg.useSkinMask   = true;
cfg.skinCb        = [77 127];
cfg.skinCr        = [133 173];
cfg.minSkinPixels = 50;     % fall back to all ROI pixels below this

% ---------------- Signal processing ----------------
cfg.hrBand        = [0.7 3.0];  % Hz (42-180 bpm)
cfg.filterOrder   = 4;          % Butterworth, zero-phase (filtfilt)
cfg.detrendLambda = 100;        % smoothness-priors detrending

cfg.posWindowSec  = 1.6;        % POS internal window
cfg.hrWindowSec   = 10;         % HR estimation window
cfg.hrStepSec     = 1;          % HR window step
cfg.nfftMin       = 2^14;       % zero-padding for finer frequency resolution

cfg.methods       = {'GREEN','CHROM','POS'};
cfg.roiSet        = {'forehead','leftCheek','rightCheek','combined'};

% ---------------- Constraint tests (Section 7 hook) ----------------
cfg.degrade = struct('enabled', false, ...
                     'lowLightGain', [], ...   % e.g. 0.2
                     'noiseType', 'none', ...  % 'none' | 'poisson' | 'gaussian'
                     'resizeTo', []);          % e.g. [240 320]

% ---------------- Figures ----------------
cfg.snippetSec = 10;   % length of pulse-vs-PPG snippet figure

% ---------------- Overrides ----------------
if mod(numel(varargin), 2) ~= 0
    error('getConfig:args', 'Overrides must be name/value pairs.');
end
for k = 1:2:numel(varargin)
    name = varargin{k};
    if ~isfield(cfg, name)
        error('getConfig:unknownField', 'Unknown config field "%s".', name);
    end
    cfg.(name) = varargin{k+1};
end
end
