function [fv, fd] = faceTraces(videoPath, subject, cfg, p, fd)
%FACETRACES Sulojan's face module on one video, cached on disk.
%   [fv, fd] = faceTraces(videoPath, subject, cfg, p, fd)
%   videoPath : UBFC vid.avi
%   subject   : e.g. 'subject3' (cache file name, metadata)
%   cfg       : getConfig struct (traceDir, maxFrames)
%   p         : face_params struct (from cfg.faceConfig)
%   fd        : faceDetector to reuse, or [] (created on first cache miss)
%   fv        : process_video output (t, rgb_<region>, rgb_face, npix,
%               occluded, quality, ... see process_video)
%
%   The cache <traceDir>/<subject>.mat is reused only when it was made with
%   the same face parameters and frame limit; otherwise the video is
%   processed again. roi_data / debug are dropped from the cache to keep
%   it small (they are only needed for debug videos).

if ~isfolder(cfg.traceDir), mkdir(cfg.traceDir); end
cacheFile = fullfile(cfg.traceDir, [char(subject) '.mat']);
key = struct('params', p, 'maxFrames', cfg.maxFrames, 'rngSeed', 0);

if isfile(cacheFile)
    S = load(cacheFile, 'fv', 'key');
    if isfield(S, 'key') && isequaln(S.key, key)
        fv = S.fv;
        fprintf('  face traces: cached (%s)\n', cacheFile);
        return
    end
    fprintf('  face traces: cache made with other settings, re-running.\n');
end

if isempty(fd)
    if exist('faceDetector', 'file') == 0 && exist('faceDetector', 'class') == 0
        error('faceTraces:noDetector', ['faceDetector (RetinaFace) not found. It needs ' ...
            'MATLAB R2025a+ and the add-on "Computer Vision Toolbox Model for ' ...
            'RetinaFace Face Detection".']);
    end
    fd = faceDetector(p.det.model);
end

fv = process_video(string(videoPath), Params=p, Detector=fd, MaxFrames=cfg.maxFrames, ...
                   Dataset="UBFC2", Subject=string(subject), Verbose=true);
fv = rmfield(fv, intersect(fieldnames(fv), {'roi_data', 'debug'}));
save(cacheFile, 'fv', 'key', '-v7');
end
