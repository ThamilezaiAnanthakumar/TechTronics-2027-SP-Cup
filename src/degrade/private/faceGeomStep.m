function [L, st] = faceGeomStep(st, I, t)
%FACEGEOMSTEP Target-face landmarks for one frame (private; used by aging / mask).
%   st = struct('lmk', [], 'smoother', [], 'prevC', [], 'prevW', [], 'last', []) on the first call.
%   MediaPipe FaceLandmarker (478 points); target = largest face in the first detection, then
%   the face nearest the previous one; 1-Euro smoothing (1 Hz, beta 0.02) as in
%   anatomicalPatchTraces. If no face is found the last landmarks are held (L = [] before the
%   first detection).
if isempty(st.lmk)
    st.lmk = MediaPipeLandmarker(); st.smoother = OneEuroFilter(1.0, 0.02, 1.0);
end
faces = st.lmk.detect(I, 1000 * t);
L = st.last;
if isempty(faces), return; end
C = cellfun(@(f) mean(f, 1), faces, 'UniformOutput', false); C = vertcat(C{:});
Wd = cellfun(@(f) max(f(:, 1)) - min(f(:, 1)), faces);
if isempty(st.prevC), [~, j] = max(Wd); else, [~, j] = min(vecnorm(C - st.prevC, 2, 2)); end
st.prevC = C(j, :); st.prevW = Wd(j);
L = reshape(st.smoother.update(reshape(faces{j}, 1, []), t), [], 2);
st.last = L;
end
