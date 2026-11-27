function T = run_benchmark(subjects, opts)
%RUN_BENCHMARK  Process every subject x condition clip and tabulate the metrics.
%   T = run_benchmark([1 3])
%   T = run_benchmark([1 3], Conditions=["Baseline" "MultiFace"], Rerun=true)
%
%   Needs the clips from make_test_clips. Trace files are reused unless Rerun=true.
%   Writes <TableDir>/face_benchmark.csv, face_benchmark_summary.csv,
%   face_bland_altman.csv and <FigDir>/bland_altman_by_condition.png.

arguments
    subjects (1,:) double {mustBeInteger, mustBePositive}
    opts.Conditions (1,:) string = ["Baseline" "LowResolutionQVGA" "MultiFace" "Masked" "Stress"]
    opts.ClipDir (1,1) string = "data/degraded"
    opts.TraceDir (1,1) string = "outputs/traces"
    opts.TableDir (1,1) string = "outputs/tables"
    opts.FigDir (1,1) string = "outputs/figures"
    opts.MaxFrames (1,1) double {mustBePositive} = Inf
    opts.Params struct = face_params()
    opts.Detector = []
    opts.Rerun (1,1) logical = false
end

p  = opts.Params;
fd = opts.Detector;
if isempty(fd), fd = faceDetector(p.det.model); end

rows = {};
W = struct("est", {}, "ref", {}, "tc", {}, "subject", {}, "condition", {});
for s = subjects
    for c = opts.Conditions
        side = fullfile(opts.ClipDir, c, "subject" + s + "_meta.mat");
        if ~isfile(side)
            warning("run_benchmark:noClip", "%s missing (run make_test_clips), skipped.", side);
            continue
        end
        L = load(side);
        clip = L.clip;
        name = trace_name(clip.dataset, clip.subject, clip.condition, clip.constraint);
        tracePath = fullfile(opts.TraceDir, name);
        if ~isfile(tracePath) || opts.Rerun
            fprintf("[trace] %s\n", name);
            out = process_video(string(clip.video_path), Params=p, Detector=fd, ...
                MaxFrames=opts.MaxFrames, Dataset=clip.dataset, Subject=clip.subject, ...
                Condition=clip.condition, Constraint=clip.constraint);
            export_traces(out, opts.TraceDir);
        end
        S  = load(tracePath);
        gt = load_ubfc_gt(string(clip.gt_path));
        [m, w] = benchmark_metrics(S, clip, gt);
        rows{end + 1, 1} = struct2table(m, "AsArray", true); %#ok<AGROW>
        W(end + 1) = w; %#ok<AGROW>
    end
end
if isempty(rows)
    T = table();
    warning("run_benchmark:empty", "Nothing was benchmarked.");
    return
end

T = vertcat(rows{:});
if ~isfolder(opts.TableDir), mkdir(opts.TableDir); end
writetable(T, fullfile(opts.TableDir, "face_benchmark.csv"));

isNum = varfun(@isnumeric, T, "OutputFormat", "uniform");
Ssum = varfun(@(x) mean(x, "omitnan"), T, "GroupingVariables", "condition", ...
              "InputVariables", T.Properties.VariableNames(isNum));
Ssum.Properties.VariableNames = erase(Ssum.Properties.VariableNames, "Fun_");
writetable(Ssum, fullfile(opts.TableDir, "face_benchmark_summary.csv"));

show = ["subject" "condition" "resolution" "target_found_pct" "id_switch_frames" ...
        "nfaces_correct_pct" "jitter_px" "occl_accuracy_pct" "ms_per_frame" "hr_mae" "suc5_pct"];
fprintf("\n=== Benchmark (per clip) ===\n");
disp(T(:, show));

BA = bland_altman(W, FigDir=opts.FigDir, TableDir=opts.TableDir);
fprintf("\n=== Bland-Altman (pooled 10 s windows) ===\n");
disp(BA(:, ["condition" "n_windows" "bias" "loa_lo" "loa_hi" "mae" "rmse" "suc3_pct" "suc5_pct"]));
fprintf("Tables: %s   Figure: %s\n", opts.TableDir, fullfile(opts.FigDir, "bland_altman_by_condition.png"));
end
