function T = run_pipeline(opts)
%RUN_PIPELINE  Test clips -> ROI traces -> benchmark -> debug videos.
%   addpath(genpath("src"))
%   T = run_pipeline(Quick=true)                  first 300 frames per video
%   T = run_pipeline()                            full videos
%   T = run_pipeline(Config="config/my.json", Subjects=[1 12], Rerun=true)
%
%   Options: Subjects, Config (JSON, default config/face_config.json), Quick,
%   MakeClips, Rerun, DebugVideos. Quick runs write to "_quick" subfolders.

arguments
    opts.Subjects (1,:) double {mustBeInteger, mustBePositive} = [1 3 4 12 17 26 31 34]
    opts.Config (1,1) string = "config/face_config.json"
    opts.Quick (1,1) logical = false
    opts.MakeClips (1,1) logical = true
    opts.Rerun (1,1) logical = false
    opts.DebugVideos (1,1) logical = true
end

oldDir = cd(repo_root());
restoreDir = onCleanup(@() cd(oldDir));

p = face_params(opts.Config);
sub = "";
maxF = Inf;
if opts.Quick
    sub = "_quick";
    maxF = 300;
end
clipDir  = fullfile("data", "degraded", sub);
traceDir = fullfile("outputs", "traces", sub);
tableDir = fullfile("outputs", "tables", sub);
figDir   = fullfile("outputs", "figures", sub);
fprintf("=== Face module: subjects %s, regions [%s], %s ===\n", mat2str(opts.Subjects), ...
        strjoin(p.roi.names, " "), opts.Config);

tAll = tic;
fd = faceDetector(p.det.model);

if opts.MakeClips
    make_test_clips(opts.Subjects, OutDir=clipDir, MaxFrames=maxF, Params=p, Detector=fd);
end

T = run_benchmark(opts.Subjects, ClipDir=clipDir, TraceDir=traceDir, TableDir=tableDir, ...
                  FigDir=figDir, MaxFrames=maxF, Params=p, Detector=fd, Rerun=opts.Rerun);

if opts.DebugVideos
    C = clip_conditions();
    s = opts.Subjects(1);
    for c = ["Baseline" "MultiFace" "Masked" "Stress"]
        name = trace_name("UBFC2", "subject" + s, c, C([C.name] == c).constraint);
        tp = fullfile(traceDir, name);
        if isfile(tp)
            debug_video(tp, OutPath=fullfile(figDir, erase(name, ".mat") + "_debug.avi"), ...
                        MaxFrames=min(maxF, 600));
        end
    end
end
fprintf("=== done in %.1f min ===\n", toc(tAll) / 60);
end
