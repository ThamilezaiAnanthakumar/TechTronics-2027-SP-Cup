function [m, w] = benchmark_metrics(S, clip, gt)
%BENCHMARK_METRICS  Face-module metrics of one processed clip.
%   [m, w] = benchmark_metrics(S, clip, gt)
%
%   S     loaded trace file,  clip  sidecar from make_test_clips,  gt  load_ubfc_gt
%   m     one table row: target found %, target IoU, ID switches, face count,
%         ROI jitter (raw / smoothed), skin pixels and occlusion % per region,
%         occlusion accuracy, ms/frame, sanity HR (MAE, RMSE, r, Bland-Altman,
%         SUC@3/5) vs contact-PPG HR
%   w     window-level HR pairs (est, ref, tc, subject, condition)

q = S.quality(:);
N = numel(q);
valid = q > 0;
meta = S.meta;
names = string(S.roi_names);
nR = numel(names);
ref = string(meta.params.occl.reference);
refIdx = find(names == ref, 1);
if isempty(refIdx), refIdx = 1; end

m.subject    = string(meta.subject);
m.condition  = string(meta.condition);
m.constraint = string(meta.constraint);
m.resolution = string(sprintf("%dx%d", meta.width, meta.height));
m.frames     = N;
m.target_found_pct = 100 * mean(valid);

% identity (C3)
faceBox = reshape([S.roi_data.face_box], 4, []).';
n = min(N, size(clip.target_box, 1));
refBox = NaN(N, 4);
refBox(1:n, :) = clip.target_box(1:n, :);
iouT = rowwise_iou(faceBox, refBox);
m.target_iou = mean(iouT(valid), "omitnan");
if ~isempty(clip.bystander_box)
    m.id_switch_frames = nnz(valid & box_iou(clip.bystander_box, faceBox) > iouT);
else
    m.id_switch_frames = 0;
end
m.target_id_changes  = nnz(diff(S.target_id(valid)) ~= 0);
m.nfaces_correct_pct = 100 * mean(S.n_faces(valid) == clip.expected_faces);

% ROI jitter of the reference region centre (motion slower than 1 s removed)
wJit = max(3, round(1 / median(diff(S.t))));
cSm = NaN(N, 2);
for k = find(valid).'
    P = S.roi_data(k).(names(refIdx));
    if ~isempty(P), cSm(k, :) = mean(P, 1); end
end
m.face_width_px   = median(faceBox(valid, 3), "omitnan");
m.jitter_raw_px   = hp_std(S.debug.ref_center_raw, valid, wJit);
m.jitter_px       = hp_std(cSm, valid, wJit);
m.jitter_pct_face = 100 * m.jitter_px / m.face_width_px;

% pixels (C4) and occlusion (C7) per region
C = clip_conditions();
expOcc = ismember(names, C([C.name] == string(clip.condition)).expected_occluded);
occ = S.occluded(valid, :) > 0;
for r = 1:nR
    m.("npix_" + names(r)) = mean(S.npix(valid, r));
end
for r = 1:nR
    m.("occl_" + names(r) + "_pct") = 100 * mean(occ(:, r));
end
m.occl_accuracy_pct = 100 * mean(occ == expOcc, "all");

m.ms_per_frame = meta.timing.ms_per_frame;
m.ms_detect    = meta.timing.ms_detect;

% sanity HR vs contact PPG
[hrGt, tc] = ppg_hr(gt.t, gt.ppg);
num = zeros(N, 3);  den = zeros(N, 1);
for r = 1:nR
    X = S.("rgb_" + names(r));
    ok = S.npix(:, r) > 0 & all(isfinite(X), 2);
    num(ok, :) = num(ok, :) + X(ok, :) .* S.npix(ok, r);
    den(ok)    = den(ok) + S.npix(ok, r);
end
naive = num ./ den;
naive(den == 0, :) = NaN;

hrEst = sanity_hr(S.t, S.rgb_face, Centers=tc);
[m.hr_mae, m.hr_rmse, m.hr_r, m.hr_windows] = hr_err(hrEst, hrGt);
m.hr_mae_naive = hr_err(sanity_hr(S.t, naive, Centers=tc), hrGt);
m.("hr_mae_" + names(refIdx)) = hr_err(sanity_hr(S.t, S.("rgb_" + names(refIdx)), Centers=tc), hrGt);

ok = isfinite(hrEst) & isfinite(hrGt);
d  = hrEst(ok) - hrGt(ok);
m.ba_bias   = mean(d);
m.ba_loa_lo = m.ba_bias - 1.96 * std(d);
m.ba_loa_hi = m.ba_bias + 1.96 * std(d);
m.suc3_pct  = 100 * mean(abs(d) < 3);
m.suc5_pct  = 100 * mean(abs(d) < 5);

w.est = hrEst(ok);
w.ref = hrGt(ok);
w.tc  = tc(ok);
w.subject   = m.subject;
w.condition = m.condition;
end

function iou = rowwise_iou(A, B)
x1 = max(A(:,1), B(:,1));  y1 = max(A(:,2), B(:,2));
x2 = min(A(:,1) + A(:,3), B(:,1) + B(:,3));
y2 = min(A(:,2) + A(:,4), B(:,2) + B(:,4));
inter = max(0, x2 - x1) .* max(0, y2 - y1);
iou = inter ./ (A(:,3) .* A(:,4) + B(:,3) .* B(:,4) - inter);
iou(~isfinite(iou)) = 0;
end

function s = hp_std(c, valid, w)
c = c(valid, :);
if size(c, 1) < 2 * w
    s = NaN;
    return
end
s = mean(std(c - movmean(c, w, 1, "omitnan"), 0, 1, "omitnan"));
end

function [mae, rmse, r, nwin] = hr_err(est, ref)
ok = isfinite(est) & isfinite(ref);
nwin = nnz(ok);
e = est(ok) - ref(ok);
mae  = mean(abs(e));
rmse = sqrt(mean(e .^ 2));
r = NaN;
if nwin > 2
    c = corrcoef(est(ok), ref(ok));
    r = c(1, 2);
end
end
