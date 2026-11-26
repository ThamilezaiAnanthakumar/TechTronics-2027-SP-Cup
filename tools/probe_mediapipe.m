%PROBE_MEDIAPIPE MATLAB -> Python MediaPipe FaceLandmarker (README 4.1 optional dense landmarks).
%   Records: pyenv, landmark count and format, per-frame time. One UBFC frame, local only.
root = fileparts(fileparts(mfilename('fullpath')));
py311 = "C:\Users\mufte\AppData\Local\Programs\Python\Python311\python.exe";
pe = pyenv;
if pe.Status == "NotLoaded", pe = pyenv(Version=py311, ExecutionMode="OutOfProcess"); end
fprintf('pyenv: %s %s (%s)\n', pe.Executable, pe.Version, pe.ExecutionMode);
addpath(fullfile(root, 'config')); addpath(genpath(fullfile(root, 'src')));
cfg = getConfig('subject', 'subject1'); [vp, ~] = loadUBFC(cfg);
v = VideoReader(vp); I = readFrame(v);
model = fullfile(root, 'models', 'face_landmarker.task');
pyrun(["import mediapipe as mp", "import numpy as np", ...
    "from mediapipe.tasks.python import vision, BaseOptions", ...
    "opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=mp_model), num_faces=4, running_mode=vision.RunningMode.IMAGE)", ...
    "lmk = vision.FaceLandmarker.create_from_options(opts)"], mp_model=char(model));
code = ["img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(frame, dtype=np.uint8)))", ...
        "res = lmk.detect(img)", ...
        "pts = np.array([[[p.x, p.y, p.z] for p in f] for f in res.face_landmarks], dtype=float)"];
t0 = tic; pts = pyrun(code, "pts", frame=py.numpy.array(I)); t1 = toc(t0);
tic; for k = 1:10, pts = pyrun(code, "pts", frame=py.numpy.array(I)); end; tAvg = toc / 10;
P = double(pts);
fprintf('faces %d, landmarks per face %d, coords per landmark %d (normalised x,y in [0,1], z relative)\n', size(P, 1), size(P, 2), size(P, 3));
fprintf('first call %.0f ms, steady %.0f ms/frame (640x480, incl. MATLAB->Python copy)\n', 1000 * t1, 1000 * tAvg);
fprintf('landmark 1 (nose tip) at (%.3f, %.3f)\n', P(1, 2, 1), P(1, 2, 2));
