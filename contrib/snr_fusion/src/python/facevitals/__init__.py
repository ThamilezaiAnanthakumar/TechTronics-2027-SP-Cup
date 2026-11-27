"""Python port of the TechTronics FaceVitals pipeline, used for the C2 (skin tone) evaluation.

face     face box, ROI placement, skin mask, per-frame RGB traces, ITA
rppg     POS, band-pass, windowed spectral HR (port of src/eval/sanity_hr.m, ppg_hr.m, spectral_hr.m)
datasets loaders for UBFC-rPPG, MPU-rPPG, CLBP-400, AiPEX rppg_biases, SRMC-N
"""
