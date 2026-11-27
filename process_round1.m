function out = process_round1(videoPath, varargin)
% PROCESS_ROUND1  FV-TT heart-rate pipeline (IEEE SP Cup 2027, Round 1).
%   out = process_round1('target_video.mp4')
%   out = process_round1(v, 'Reference','reference_signal.csv', 'Mode','offline', ...
%                        'Config','config/default.json', 'OutDir','Round1_Baseline', ...
%                        'Constraint','None', 'Condition','Baseline', 'Params',paramsStruct)
% Writes processed_output.csv and, if a reference is given, RoundX_Condition_Constraint.mat
% with facial_video, roi_data, reference_signal, processed_signal, metrics, params.
%
% Name/value options
%   Reference       '' | CSV path (columns time_s and ppg and/or hr_bpm) | struct (t, ppg,
%                   hrSensor or hr_bpm). The reference HR is computed with the plain chain
%                   (README 4.10), never with the tracker.
%   Mode            'offline' (only mode implemented in MATLAB; 'online' -> error)
%   Config          '' (defaults: getConfig with filterOrder 2) | struct | JSON file whose
%                   top-level fields override getConfig defaults
%   OutDir          output folder (default: ./Round1_<Condition>)
%   Constraint      'None', 'C1' ... 'C7'          Condition  'Baseline', 'LowLight', ...
%   Params          struct merged into the saved params (e.g. lux, distance_cm, sidecar)
%   MaxFrames       process only the first N frames (default Inf)
%   MaxFramesInMat  frames of facial_video stored in the .mat (default Inf = all; 0 = none).
%                   facial_video is ~1-2 GB per VGA minute (organiser question pending).
%   Tracker         'none' (default, per-window spectral fusion) | 'viterbi'
%   Round           'Round1' (file name prefix)
%   Traces          precomputed patch traces (anatomicalPatchTraces struct; tests only)
%
% Output struct: processed_signal, roi_data, metrics, params, csvFile, matFile.
% Requires: MATLAB R2026b, Image Processing, Signal Processing, Computer Vision Toolboxes;
% Python 3.11 + mediapipe for the landmarks (see docs/API_NOTES.md).

ip = inputParser;
ip.addRequired('videoPath', @(x) ischar(x) || isstring(x));
ip.addParameter('Reference', '');
ip.addParameter('Mode', 'offline', @(x) any(strcmpi(x, {'offline', 'online'})));
ip.addParameter('Config', '');
ip.addParameter('OutDir', '');
ip.addParameter('Constraint', 'None');
ip.addParameter('Condition', 'Baseline');
ip.addParameter('Params', struct(), @isstruct);
ip.addParameter('MaxFrames', Inf, @isnumeric);
ip.addParameter('MaxFramesInMat', Inf, @isnumeric);
ip.addParameter('Tracker', 'none');
ip.addParameter('Round', 'Round1');
ip.addParameter('Traces', struct(), @isstruct);
ip.parse(videoPath, varargin{:});
o = ip.Results;
videoPath = char(o.videoPath);

root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'config')); addpath(genpath(fullfile(root, 'src')));
if strcmpi(o.Mode, 'online')
    error('process_round1:mode', 'Mode ''online'' is not implemented in MATLAB yet; use ''offline''.');
end
cfg = loadConfig(o.Config);
outDir = char(o.OutDir);
if isempty(outDir), outDir = fullfile(pwd, ['Round1_' char(o.Condition)]); end
if ~isfolder(outDir), mkdir(outDir); end

ref = loadReference(o.Reference);
R = runOffline(videoPath, Reference=ref, Config=cfg, MaxFrames=o.MaxFrames, ...
    Traces=o.Traces, Tracker=string(o.Tracker));
ps = R.processed_signal;

% ---- processed_output.csv (README 4.9) ----------------------------------------------
csvFile = fullfile(outDir, 'processed_output.csv');
Tcsv = table(round(ps.t, 3), round(ps.hr_bpm, 2), round(ps.confidence, 3), string(ps.status), ...
    'VariableNames', {'time_s', 'hr_bpm', 'confidence', 'status'});
writetable(Tcsv, csvFile);

% ---- params --------------------------------------------------------------------------
vi = videoProps(videoPath, R);
params = struct('resolution', [vi.width vi.height], 'fps', R.info.fps_measured, ...
    'fps_nominal', vi.fps_nominal, 'lux', NaN, 'distance_cm', NaN, 'fitzpatrick', '', ...
    'ita_deg', R.info.ita_deg, 'age', NaN, 'constraints_applied', char(o.Constraint), ...
    'condition', char(o.Condition), 'pipeline_version', pipelineVersion(), 'git_hash', gitInfo(), ...
    'tracker', char(o.Tracker), 'mode', 'offline', 'video', videoPath, ...
    'frames_processed', R.info.num_frames, 'fs_proc', R.info.fs_proc, ...
    'window_protocol', sprintf('%g s windows, %g s step', cfg.hrWindowSec, cfg.hrStepSec), ...
    'config', struct('filterOrder', cfg.filterOrder, 'hrBand', cfg.hrBand, ...
        'detrendLambda', cfg.detrendLambda, 'posWindowSec', cfg.posWindowSec, ...
        'minValidFrac', cfg.minValidFrac, 'topK', 5, 'lowSnrDb', -3, 'maskedFrac', 0.4), ...
    'matlab', version, 'created', char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss')));
f = fieldnames(o.Params);
for k = 1:numel(f), params.(f{k}) = o.Params.(f{k}); end

out = struct('processed_signal', ps, 'roi_data', R.roi_data, 'metrics', R.metrics, ...
    'params', params, 'csvFile', csvFile, 'matFile', '');

% ---- RoundX_Condition_Constraint.mat (README 1.3), only with a reference --------------
if ~isempty(ref)
    processed_signal = ps; roi_data = R.roi_data; metrics = R.metrics; %#ok<NASGU>
    reference_signal = struct('t', ref.t(:), 'ppg', ref.ppg(:), 'hr_bpm', ref.hr_bpm(:), ...
        'source', ref.source, 't_win', R.reference.t_win, 'hr_win', R.reference.hr_win, ...
        'hr_win_method', R.reference.source); %#ok<NASGU>
    [facial_video, nKept] = readFrames(videoPath, min(o.MaxFramesInMat, R.info.num_frames));
    params.facial_video_frames = nKept;
    params.facial_video_note = sprintf('first %d of %d processed frames stored (MaxFramesInMat)', nKept, R.info.num_frames);
    matFile = fullfile(outDir, sprintf('%s_%s_%s.mat', o.Round, o.Condition, o.Constraint));
    save(matFile, 'facial_video', 'roi_data', 'reference_signal', 'processed_signal', 'metrics', 'params', '-v7.3');
    out.params = params; out.matFile = matFile;
end
end

% =====================================================================================
function cfg = loadConfig(c)
if isstruct(c), cfg = c; return; end
cfg = getConfig('filterOrder', 2);
if ~isempty(c)
    J = jsondecode(fileread(c)); f = fieldnames(J);
    for k = 1:numel(f), cfg.(f{k}) = J.(f{k}); end
end
end

function ref = loadReference(r)
% -> [] or struct t, ppg, hr_bpm, hrSensor, source (column vectors; missing -> NaN)
ref = [];
if isempty(r), return; end
if ischar(r) || isstring(r)
    T = readtable(r); src = char(r);
    g.t = T.time_s;
    if ismember('ppg', T.Properties.VariableNames), g.ppg = T.ppg; end
    if ismember('hr_bpm', T.Properties.VariableNames), g.hr_bpm = T.hr_bpm; end
else
    g = r; src = 'struct';
    if isfield(g, 'path'), src = g.path; end
end
n = numel(g.t);
ref.t = g.t(:);
ref.ppg = NaN(n, 1); if isfield(g, 'ppg'), ref.ppg = g.ppg(:); end
if isfield(g, 'hr_bpm'), ref.hr_bpm = g.hr_bpm(:);
elseif isfield(g, 'hrSensor'), ref.hr_bpm = g.hrSensor(:);
else, ref.hr_bpm = NaN(n, 1); end
ref.hrSensor = ref.hr_bpm;
ref.source = char(src);
end

function vi = videoProps(videoPath, R)
vi = struct('width', NaN, 'height', NaN, 'fps_nominal', NaN);
if isfile(videoPath)
    v = VideoReader(videoPath); vi = struct('width', v.Width, 'height', v.Height, 'fps_nominal', v.FrameRate);
elseif isfield(R.roi_data, 'frame_size')
    vi.width = R.roi_data.frame_size(2); vi.height = R.roi_data.frame_size(1);
end
end

function [V, k] = readFrames(videoPath, n)
V = zeros(0, 0, 3, 0, 'uint8'); k = 0;
if n <= 0 || ~isfile(videoPath), return; end
v = VideoReader(videoPath);
n = min(n, v.NumFrames);
V = zeros(v.Height, v.Width, 3, n, 'uint8');
while hasFrame(v) && k < n, k = k + 1; V(:, :, :, k) = readFrame(v); end
V = V(:, :, :, 1:k);
end

function s = pipelineVersion()
s = ['FV-TT round1 0.1: MediaPipe 478-pt landmarks -> 10 anatomical patches (shared/patches.json), ' ...
     '1-Euro, skin+specular mask, trimmed mean -> POS -> detrend + Butterworth order 2 (0.7-3 Hz) -> ' ...
     '10 s / 1 s windows -> per-window SNR-weighted top-5 spectral fusion'];
end
