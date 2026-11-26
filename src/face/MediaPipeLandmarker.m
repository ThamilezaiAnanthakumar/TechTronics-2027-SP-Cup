classdef MediaPipeLandmarker < handle
%MEDIAPIPELANDMARKER MediaPipe FaceLandmarker (478 landmarks) called from MATLAB via Python.
%   lm = MediaPipeLandmarker();             model: <repo>/models/face_landmarker.task
%   faces = lm.detect(I, t_ms)              I: HxWx3 uint8 RGB, t_ms: increasing integer timestamp
%   faces: 1xF cell, each 478x2 double, pixel coordinates (x, y), 1-based like MATLAB images.
%   Uses VIDEO running mode (temporal tracking inside MediaPipe), up to 4 faces.
%   Needs Python 3.11 with mediapipe (docs/API_NOTES.md); pyenv is set OutOfProcess on first use.
    properties (Constant)
        Python = "C:\Users\mufte\AppData\Local\Programs\Python\Python311\python.exe"
    end
    properties (SetAccess = private)
        ModelPath string
        LastT = -Inf
    end
    methods
        function obj = MediaPipeLandmarker(modelPath)
            if nargin < 1
                root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
                modelPath = fullfile(root, 'models', 'face_landmarker.task');
            end
            obj.ModelPath = string(modelPath);
            pe = pyenv;
            if pe.Status == "NotLoaded"
                pyenv(Version=MediaPipeLandmarker.Python, ExecutionMode="OutOfProcess");
            end
            pyrun(["import mediapipe as mp", "import numpy as np", ...
                "from mediapipe.tasks.python import vision, BaseOptions", ...
                "fv_opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=mp_model), num_faces=4, running_mode=vision.RunningMode.VIDEO)", ...
                "fv_lmk = vision.FaceLandmarker.create_from_options(fv_opts)"], mp_model=char(obj.ModelPath));
        end

        function faces = detect(obj, I, t_ms)
            t_ms = max(round(t_ms), obj.LastT + 1); obj.LastT = t_ms;
            code = ["img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(frame, dtype=np.uint8)))", ...
                    "res = fv_lmk.detect_for_video(img, int(ts))", ...
                    "pts = np.array([[[p.x, p.y] for p in f] for f in res.face_landmarks], dtype=float).reshape(-1, 478, 2)"];
            P = double(pyrun(code, "pts", frame=py.numpy.array(I), ts=int64(t_ms)));
            [H, W, ~] = size(I);
            faces = cell(1, size(P, 1));
            for k = 1:size(P, 1)
                xy = reshape(P(k, :, :), 478, 2);
                faces{k} = [xy(:, 1) * W + 0.5, xy(:, 2) * H + 0.5];   % normalised -> 1-based pixels
            end
        end
    end
end
