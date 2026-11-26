function make_report_figures()
%MAKE_REPORT_FIGURES Report figures from the committed result tables (outputs/repro).
%   Writes docs/report/figures/*.pdf. Every plotted number comes from:
%     outputs/repro/ubfc2_seeded_per_subject.csv, ubfc2_seeded_pooled.csv (run_ubfc_all @ 318426d)
%     outputs/repro/aipex_muftee_metrics_avgspec.csv (run_aipex @ f9d4155)
%     published AiPEX baselines: Dasari et al. 2021 (Results_Final_Main.csv in the dataset)
root = fileparts(fileparts(mfilename('fullpath')));
R = fullfile(root, 'outputs', 'repro');
F = fullfile(root, 'docs', 'report', 'figures');
if ~isfolder(F), mkdir(F); end
blue = [0.165 0.471 0.839]; orange = [0.922 0.408 0.204]; grey = [0.62 0.62 0.60];

% --- Fig. 1: per-subject MAE, POS / combined, UBFC2 (42) ---
P = readtable(fullfile(R, 'ubfc2_seeded_per_subject.csv'), 'TextType', 'string');
P = P(P.Method == "POS" & P.ROI == "combined", :);
[mae, i] = sort(P.MAE);  subj = erase(P.Subject(i), "subject");
f = figure('Visible', 'off', 'Units', 'centimeters', 'Position', [0 0 17 5.2]);
if exist('theme', 'file'), theme(f, 'light'); end   % R2025a+ defaults to the desktop (dark) theme
b = bar(min(mae, 8), 0.75, 'FaceColor', 'flat', 'EdgeColor', 'none');
b.CData = repmat(blue, numel(mae), 1);  b.CData(mae > 5, :) = repmat(orange, nnz(mae > 5), 1);
hold on; yline(median(mae), '--', sprintf('median %.2f bpm', median(mae)), 'Color', [0.3 0.3 0.3], ...
    'LabelHorizontalAlignment', 'left', 'FontSize', 7);
for k = find(mae > 8)', text(k, 8.2, sprintf('%.1f', mae(k)), 'HorizontalAlignment', 'center', 'FontSize', 7); end
set(gca, 'XTick', 1:numel(mae), 'XTickLabel', subj, 'FontSize', 6.5, 'YLim', [0 9], 'Box', 'off');
xlabel('UBFC-rPPG subject (sorted)'); ylabel('MAE (bpm)'); grid on; ax = gca; ax.XGrid = 'off';
exportgraphics(f, fullfile(F, 'ubfc2_per_subject.pdf'), 'ContentType', 'vector'); close(f);

% --- Fig. 2: AiPEX, ours vs published (all 88 videos) ---
A = readtable(fullfile(R, 'aipex_muftee_metrics_avgspec.csv'), 'TextType', 'string');
A = A(A.Subset == "All" & A.Method ~= "GREEN", :);
[~, j] = min(A.MAE);
B = readtable(fullfile(R, 'aipex_paper_baseline.csv'), 'TextType', 'string');   % from the dataset's Results_Final_Main.csv
B = B(B.Subset == "All", :);
label = containers.Map(["BKF" "Chrom" "POS" "Sph" "DP"], ["BKF" "CHROM" "POS" "Sph. mean" "DeepPhys"]);
names = ["Published " + arrayfun(@(m) string(label(m)), B.Method)', ...
         "Ours: " + A.Method(j) + " / " + A.ROI(j), "Ours: POS / combined"];
vals  = [B.MAE', A.MAE(j), A.MAE(A.Method == "POS" & A.ROI == "combined")];
ours  = [false(1, height(B)), true, true];
[vals, o] = sort(vals); names = names(o); ours = ours(o);
f = figure('Visible', 'off', 'Units', 'centimeters', 'Position', [0 0 8.6 5.4]);
if exist('theme', 'file'), theme(f, 'light'); end   % R2025a+ defaults to the desktop (dark) theme
b = barh(vals, 0.65, 'FaceColor', 'flat', 'EdgeColor', 'none');
b.CData = repmat(grey, numel(vals), 1); b.CData(ours, :) = repmat(blue, nnz(ours), 1);
for k = 1:numel(vals), text(vals(k) + 0.3, k, sprintf('%.1f', vals(k)), 'FontSize', 7); end
set(gca, 'YTick', 1:numel(vals), 'YTickLabel', names, 'FontSize', 7, 'XLim', [0 25], 'Box', 'off');
xlabel('Video-level MAE (bpm), 88 videos'); grid on; ax = gca; ax.YGrid = 'off';
exportgraphics(f, fullfile(F, 'aipex_comparison.pdf'), 'ContentType', 'vector'); close(f);
fprintf('figures written to %s\n', F);
end
