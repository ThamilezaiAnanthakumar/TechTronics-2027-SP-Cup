TechTronics — IEEE SP Cup 2027, Round 1 technical report (DRAFT, 27 Nov 2026)

Build
  Overleaf: upload the zip (New Project -> Upload Project) or replace main.tex, refs.bib and figures/
  in the existing project; compiler pdfLaTeX, main file main.tex.
  Local:    pdflatex main && bibtex main && pdflatex main && pdflatex main
  Needs IEEEtran, siunitx, booktabs, tikz. Not compiled on the author's machine (no LaTeX installed);
  static checks only (balanced braces/environments, all \cite in refs.bib, all \ref labelled, figures present).

Where the numbers come from (repository TechTronics-2027-SP-Cup, branch dev; frozen config tag round1-freeze)
  Table II (UBFC)      outputs/repro/ubfc2_seeded_pooled.csv (318426d); outputs/ablation_roi/summary.csv (a275573);
                       outputs/round1/ubfc_dev_summary.csv (tools/eval_ubfc_cp2.m, 16ec481)
  Fig. 2               tools/make_report_figures.m from outputs/repro/ubfc2_seeded_per_subject.csv
  Tables III           face module run_pipeline @ 921910d, reproduced 26 Nov (docs/REPRODUCTION.md)
  Table IV (params)    config/round1_freeze.json
  Fig. 3 (waterfall)   outputs/aipex_iter/log.csv (tools/aipex_iter.m), make_report_figures.m
  Table VI (AiPEX)     outputs/round1/aipex_test_metrics.csv, aipex_dev_metrics.csv (tools/aipex_metrics.py, 30a2266)
  SNR fusion (Nadha)   docs/SNR_FUSION.md, outputs/snr_fusion/ (reproduced exactly; merge fd87348)
  Fig. 4 + C2 text     outputs/round1/c2_dev_E019_*.csv, c2_test_CP2_*.csv (tools/c2_aipex.py, tools/make_c2_figure.py)
  MP4 encoding         docs/API_NOTES.md (tools/probe_mpeg4_quality.m, tools/probe_ffmpeg_mp4.m)

Placeholders (red TBD / [TODO] in the PDF), to be filled when the runs finish
  - UBFC test split (30 subjects): Table II last row, abstract, discussion   <- tools/eval_ubfc_cp2.m('test')
  - Constraint suite (Table V): 4 test subjects x 30 s, C1/C4/C5/C6/C7      <- tools/run_constraints.m
  - Repeated-measures LoA and subject-level bootstrap CIs for UBFC
  - Web demonstrator vs MATLAB agreement, timing, screenshots
  - Contributions of Rajinthan Rameshkumar, John Praveen, Joshua Kulasingham; supervisor; licence
  - Three references from README Section 19 with incomplete records (li2026roi, bondarenko2025roi, tian2026correction)
  - Related work: organiser group's prior work, aging-skin reference

Privacy: no Sri Ramachandra data or faces are included.
