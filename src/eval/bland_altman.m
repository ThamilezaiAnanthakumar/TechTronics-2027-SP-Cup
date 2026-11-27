function R = bland_altman(W, opts)
%BLAND_ALTMAN  Pooled Bland-Altman analysis per condition, with a figure.
%   R = bland_altman(W, FigDir="outputs/figures", TableDir="outputs/tables")
%   W  window-level HR pairs from benchmark_metrics (one element per clip)
%   R  one row per condition: bias (95 % CI), limits of agreement bias -/+ 1.96 SD
%      (95 % CI), trend r, MAE, RMSE, Pearson r, SUC@3/5, per-subject-mean stats.
%   Altman & Bland, The Statistician 1983; CIs from Bland & Altman, Lancet 1986.

arguments
    W struct
    opts.FigDir (1,1) string = "outputs/figures"
    opts.TableDir (1,1) string = "outputs/tables"
end

conds = unique([W.condition], "stable");
subjAll = unique([W.subject], "stable");
nC = numel(conds);
rows = cell(nC, 1);

fig = figure("Visible", "off", "Position", [50 50 480 * min(nC, 3) 400 * ceil(nC / 3)]);
theme(fig, "light");
tl = tiledlayout(fig, ceil(nC / 3), min(nC, 3), "TileSpacing", "compact", "Padding", "compact");
cmap = lines(numel(subjAll));

for c = 1:nC
    Wc = W([W.condition] == conds(c));
    est = vertcat(Wc.est);  ref = vertcat(Wc.ref);
    sIdx = repelem((1:numel(Wc)).', arrayfun(@(x) numel(x.est), Wc));
    d = est - ref;  mn = (est + ref) / 2;  n = numel(d);

    s.condition  = conds(c);
    s.n_subjects = numel(Wc);
    s.n_windows  = n;
    s.bias       = mean(d);
    sd           = std(d);
    s.sd         = sd;
    s.bias_ci_lo = s.bias - 1.96 * sd / sqrt(n);
    s.bias_ci_hi = s.bias + 1.96 * sd / sqrt(n);
    s.loa_lo     = s.bias - 1.96 * sd;
    s.loa_hi     = s.bias + 1.96 * sd;
    s.loa_ci     = 1.96 * sqrt(3 * sd^2 / n);   % +/- half-width of the CI of each limit
    [cr, cp]     = corrcoef(d, mn);
    s.trend_r    = cr(1, 2);
    s.trend_p    = cp(1, 2);
    s.mae        = mean(abs(d));
    s.rmse       = sqrt(mean(d .^ 2));
    rr           = corrcoef(est, ref);
    s.pearson_r  = rr(1, 2);
    s.suc3_pct   = 100 * mean(abs(d) < 3);
    s.suc5_pct   = 100 * mean(abs(d) < 5);

    me = arrayfun(@(x) mean(x.est), Wc);  mr = arrayfun(@(x) mean(x.ref), Wc);
    ds = me(:) - mr(:);
    s.subj_bias   = mean(ds);
    s.subj_loa_lo = mean(ds) - 1.96 * std(ds);
    s.subj_loa_hi = mean(ds) + 1.96 * std(ds);
    rows{c} = struct2table(s, "AsArray", true);

    ax = nexttile(tl);
    hold(ax, "on");
    [~, sGlobal] = ismember([Wc.subject], subjAll);
    scatter(ax, mn, d, 10, cmap(sGlobal(sIdx), :), "filled", "MarkerFaceAlpha", 0.5);
    plot(ax, (me + mr) / 2, ds, "kd", "MarkerFaceColor", "k", "MarkerSize", 6);
    yline(ax, s.bias, "-", sprintf("bias %.2f", s.bias), "LineWidth", 1.2);
    yline(ax, s.loa_lo, "--", sprintf("-1.96 SD %.2f", s.loa_lo), "LabelVerticalAlignment", "bottom");
    yline(ax, s.loa_hi, "--", sprintf("+1.96 SD %.2f", s.loa_hi));
    yline(ax, 0, ":", "Color", [0.5 0.5 0.5]);
    grid(ax, "on");
    xlabel(ax, "mean of estimate and reference (bpm)");
    ylabel(ax, "estimate - reference (bpm)");
    title(ax, sprintf("%s  (n=%d windows, %d subjects)", conds(c), n, numel(Wc)), "Interpreter", "none");
end
title(tl, "Bland-Altman: sanity HR (POS on rgb\_face) vs contact-PPG HR; dots = 10 s windows coloured by subject, diamonds = subject means");

R = vertcat(rows{:});
if ~isfolder(opts.TableDir), mkdir(opts.TableDir); end
if ~isfolder(opts.FigDir), mkdir(opts.FigDir); end
writetable(R, fullfile(opts.TableDir, "face_bland_altman.csv"));
exportgraphics(fig, fullfile(opts.FigDir, "bland_altman_by_condition.png"), "Resolution", 150);
close(fig);
end
