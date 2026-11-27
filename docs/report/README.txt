TechTronics — IEEE SP Cup 2027, Round 1 technical report (DRAFT, 26 Nov 2026)

Build
  Overleaf: upload the zip, set main.tex as the main file, compiler pdfLaTeX.
  Local:    pdflatex main && bibtex main && pdflatex main && pdflatex main
  Needs the IEEEtran class (in TeX Live / MiKTeX / Overleaf), siunitx, booktabs, tikz.
  Not compiled on the author's machine (no LaTeX installed); static checks only
  (balanced braces/environments, every \cite in refs.bib, every \ref labelled, figures present).

Where the numbers come from (repository TechTronics-2027-SP-Cup, branch dev)
  Table II, Fig. 2   outputs/repro/ubfc2_seeded_pooled.csv / _per_subject.csv   run_ubfc_all @ 318426d
  Table III          Sulojan run_pipeline @ 921910d, reproduced 26 Nov (docs/REPRODUCTION.md)
  Fig. 3             outputs/repro/aipex_muftee_metrics_avgspec.csv (run_aipex @ f9d4155)
                     + aipex_paper_baseline.csv (from the dataset's Results_Final_Main.csv)
  Figures            regenerate with tools/make_report_figures.m (MATLAB R2026b)

Open items are marked [TODO] or TBD in red: UBFC test-split numbers, the constraint table (Table V), repeated-measures LoA, web-vs-MATLAB agreement, contributions of three team members, and three references from README Section 19 still to verify (li2026roi, bondarenko2025roi, tian2026correction).
AiPEX-dev iteration and the single AiPEX-test run, failure analysis of subjects 15/27/41/49,
repeated-measures LoA and bootstrap CIs, web-demo validation, three references from README Section 19
whose full records still need checking (li2026roi, bondarenko2025roi, tian2026correction).

Privacy: no Sri Ramachandra data or faces are included.
