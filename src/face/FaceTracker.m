classdef FaceTracker < handle
%FACETRACKER  Multi-face tracking with one designated target face.
%   tracker = FaceTracker(fd, p, [H W])
%   st      = tracker.step(I, t)      call once per frame, in order
%
%   All faces: RetinaFace every p.det.every frames, Kalman filter per track,
%   Hungarian assignment on 1 - IoU. Target: chosen once by p.trk.targetPolicy,
%   followed every frame by KLT + similarity transform, re-anchored on the
%   detection when it drifts. On loss it is re-acquired only near its last
%   position (or by policy after p.trk.reacqAnyAfter seconds).
%
%   st fields: quality (0 lost, 1 tracked, 2 confirmed by detector), targetId,
%   targetBox, anchorBox, A (anchor -> current transform), boxes / ids of all
%   visible faces, nFaces, detected, numPts.

    properties (SetAccess = private)
        Params
        Detector
        FrameSize
        Tracks
        NextId        = 1
        TargetId      = NaN
        LastTargetBox = []
        Lost          = true
        LostSince     = -Inf
        SinceDet      = Inf
        PT            = []
        Pts           = zeros(0, 2)
        NumPtsInit    = 0
        AnchorBox     = NaN(1, 4)
        A             = eye(3)
        NumDetections = 0
        DetectTime    = 0
    end

    methods
        function obj = FaceTracker(detector, p, frameSize)
            obj.Detector  = detector;
            obj.Params    = p;
            obj.FrameSize = frameSize(1:2);
            obj.Tracks    = FaceTracker.emptyTracks();
        end

        function st = step(obj, I, t)
            p = obj.Params;
            G = im2gray(I);
            confirmed = false;
            detected  = false;

            % move the target with KLT; on failure search near its last position
            if ~obj.Lost && ~obj.kltUpdate(G)
                win = expand_box(obj.LastTargetBox, p.det.searchScale, obj.FrameSize);
                B = obj.runDetector(I, win);
                detected = true;
                j = pick_near(B, obj.LastTargetBox, p.trk.reacqDist);
                if j > 0
                    obj.anchor(G, B(j, :));
                    obj.setTargetTrackBox(B(j, :), true);
                    confirmed = true;
                else
                    obj.loseTarget(t);
                end
            end
            if ~obj.Lost
                obj.setTargetTrackBox(obj.currentTargetBox(), false);
            end

            % periodic full-frame detection
            period = p.det.every;
            if obj.Lost, period = p.det.everyLost; end
            obj.SinceDet = obj.SinceDet + 1;
            if obj.SinceDet >= period
                obj.SinceDet = 0;
                B = obj.runDetector(I, []);
                detected = true;
                obj.updateTracks(B);
                ti = obj.targetIndex();
                if ~isempty(ti) && obj.Tracks(ti).misses == 0
                    db = obj.Tracks(ti).det;
                    if obj.Lost || box_iou(obj.currentTargetBox(), db) < p.klt.reanchorIoU
                        obj.anchor(G, db);
                    end
                    confirmed = true;
                elseif obj.Lost
                    j = obj.chooseTarget(t);
                    if j > 0
                        obj.TargetId = obj.Tracks(j).id;
                        obj.anchor(G, obj.Tracks(j).det);
                        confirmed = true;
                    end
                end
            end

            st.detected = detected;
            st.numPts   = size(obj.Pts, 1);
            if obj.Lost
                st.quality   = 0;
                st.targetId  = NaN;
                st.targetBox = NaN(1, 4);
                st.anchorBox = NaN(1, 4);
                st.A         = eye(3);
            else
                st.quality   = 1 + double(confirmed);
                st.targetId  = obj.TargetId;
                st.targetBox = obj.currentTargetBox();
                st.anchorBox = obj.AnchorBox;
                st.A         = obj.A;
            end
            [st.boxes, st.ids] = obj.visibleFaces();
            st.nFaces = numel(st.ids);
        end
    end

    methods (Access = private)
        function B = runDetector(obj, I, roi)
            t0 = tic;
            B = detect_faces(obj.Detector, I, obj.Params, roi);
            obj.DetectTime    = obj.DetectTime + toc(t0);
            obj.NumDetections = obj.NumDetections + 1;
        end

        function updateTracks(obj, B)
            p  = obj.Params;
            nT = numel(obj.Tracks);
            nD = size(B, 1);
            ti = obj.targetIndex();

            for i = 1:nT
                c = predict(obj.Tracks(i).kf);
                if ~(isequal(i, ti) && ~obj.Lost)          % tracked target keeps its KLT box
                    b = obj.Tracks(i).box;
                    obj.Tracks(i).box = [c(1) - b(3)/2, c(2) - b(4)/2, b(3), b(4)];
                end
            end

            if nT > 0 && nD > 0
                gate = 1e3;
                cost = zeros(nT, nD);
                for i = 1:nT
                    iou = box_iou(obj.Tracks(i).box, B);
                    c = 1 - iou;
                    c(iou < p.trk.minIoU) = gate;
                    cost(i, :) = c.';
                end
                [asg, uT, uD] = assignDetectionsToTracks(cost, p.trk.costNonAssign);
                asg = double(asg);  uT = double(uT(:));  uD = double(uD(:));
                if ~isempty(asg)
                    bad = cost(sub2ind(size(cost), asg(:, 1), asg(:, 2))) >= gate;
                    uT = [uT; asg(bad, 1)];
                    uD = [uD; asg(bad, 2)];
                    asg(bad, :) = [];
                end
            else
                asg = zeros(0, 2);  uT = (1:nT).';  uD = (1:nD).';
            end

            for r = 1:size(asg, 1)
                i = asg(r, 1);
                d = B(asg(r, 2), :);
                correct(obj.Tracks(i).kf, box_center(d));
                obj.Tracks(i).box    = d;
                obj.Tracks(i).det    = d;
                obj.Tracks(i).misses = 0;
                obj.Tracks(i).hits   = obj.Tracks(i).hits + 1;
            end
            for i = uT.'
                obj.Tracks(i).misses = obj.Tracks(i).misses + 1;
                obj.Tracks(i).det    = NaN(1, 4);
            end

            del = [obj.Tracks.misses] > p.trk.maxMisses;
            if ~isempty(ti) && ~obj.Lost
                del(ti) = false;
            end
            obj.Tracks(del) = [];

            for d = uD.'
                obj.Tracks(end + 1) = obj.newTrack(B(d, :));
            end
        end

        function tr = newTrack(obj, box)
            p = obj.Params;
            tr = FaceTracker.emptyTracks();
            tr(1).id     = obj.NextId;
            tr(1).box    = box;
            tr(1).det    = box;
            tr(1).kf     = configureKalmanFilter("ConstantVelocity", box_center(box), ...
                               p.trk.kfInitErr, p.trk.kfMotionNoise, p.trk.kfMeasNoise);
            tr(1).misses = 0;
            tr(1).hits   = 1;
            obj.NextId = obj.NextId + 1;
        end

        function ti = targetIndex(obj)
            if isempty(obj.Tracks) || isnan(obj.TargetId)
                ti = [];
            else
                ti = find([obj.Tracks.id] == obj.TargetId, 1);
            end
        end

        function setTargetTrackBox(obj, box, correctKalman)
            ti = obj.targetIndex();
            if isempty(ti), return; end
            obj.Tracks(ti).box = box;
            if correctKalman
                correct(obj.Tracks(ti).kf, box_center(box));
                obj.Tracks(ti).misses = 0;
            end
        end

        function j = chooseTarget(obj, t)
            % index into obj.Tracks of the face to (re)designate as target, 0 = none
            p = obj.Params;
            j = 0;
            vis = find([obj.Tracks.misses] == 0);
            if isempty(vis), return; end
            boxes = vertcat(obj.Tracks(vis).det);

            if (t - obj.LostSince) <= p.trk.reacqAnyAfter && ~isempty(obj.LastTargetBox)
                k = pick_near(boxes, obj.LastTargetBox, p.trk.reacqDist);
                if k > 0, j = vis(k); end
                return
            end
            if isnan(obj.TargetId) && ~isempty(p.trk.targetBox)
                [best, k] = max(box_iou(p.trk.targetBox, boxes));
                if best > 0, j = vis(k); end
                return
            end
            area  = boxes(:, 3) .* boxes(:, 4);
            imC   = [obj.FrameSize(2), obj.FrameSize(1)] / 2;
            dist  = vecnorm(box_center(boxes) - imC, 2, 2) / norm(imC);
            [~, k] = max(area / max(area) - p.trk.centralWeight * dist);
            j = vis(k);
        end

        function [boxes, ids] = visibleFaces(obj)
            boxes = zeros(0, 4);  ids = zeros(0, 1);
            ti = obj.targetIndex();
            for i = 1:numel(obj.Tracks)
                isTgt = isequal(i, ti) && ~obj.Lost;
                if obj.Tracks(i).misses == 0 || isTgt
                    b = obj.Tracks(i).box;
                    if isTgt, b = obj.currentTargetBox(); end
                    boxes(end + 1, :) = b;                  %#ok<AGROW>
                    ids(end + 1, 1)   = obj.Tracks(i).id;   %#ok<AGROW>
                end
            end
        end

        function anchor(obj, G, box)
            box = clip_box(box, obj.FrameSize);
            obj.AnchorBox     = box;
            obj.A             = eye(3);
            obj.Lost          = false;
            obj.LastTargetBox = box;
            obj.initPoints(G, box);
        end

        function initPoints(obj, G, box)
            p = obj.Params;
            ins = p.klt.featInset;
            r = [box(1) + ins(1) * box(3), box(2) + ins(2) * box(4), ...
                 box(3) * (1 - 2 * ins(1)), box(4) * (1 - 2 * ins(2))];
            r = clip_box(r, obj.FrameSize, true);
            loc = zeros(0, 2);
            if all(r(3:4) >= 4)
                c = detectMinEigenFeatures(G, "ROI", r, "MinQuality", p.klt.minQuality);
                c = selectStrongest(c, p.klt.maxPts);
                loc = double(c.Location);
            end
            obj.Pts        = loc;
            obj.NumPtsInit = size(loc, 1);
            if size(loc, 1) >= p.klt.minPts
                if isempty(obj.PT)
                    obj.PT = vision.PointTracker("MaxBidirectionalError", p.klt.maxBidirErr);
                else
                    release(obj.PT);
                end
                initialize(obj.PT, loc, G);
            end
        end

        function ok = kltUpdate(obj, G)
            p = obj.Params;
            ok = false;
            if size(obj.Pts, 1) < p.klt.minPts, return; end
            [pts, valid] = step(obj.PT, G);
            old = obj.Pts(valid, :);
            new = double(pts(valid, :));
            if size(new, 1) < p.klt.minPts, return; end
            try
                [tf, inl, status] = estgeotform2d(old, new, "similarity", "MaxDistance", p.klt.maxDist);
            catch
                return
            end
            if status ~= 0 || nnz(inl) < p.klt.minPts, return; end
            Anew = tf.A * obj.A;
            s = sqrt(abs(det(Anew(1:2, 1:2))));
            if s < 0.5 || s > 2, return; end
            obj.A   = Anew;
            obj.Pts = new(inl, :);
            setPoints(obj.PT, obj.Pts);
            obj.LastTargetBox = obj.currentTargetBox();
            if size(obj.Pts, 1) < p.klt.refreshFrac * obj.NumPtsInit
                obj.initPoints(G, obj.LastTargetBox);
            end
            ok = true;
        end

        function box = currentTargetBox(obj)
            b = obj.AnchorBox;
            P = apply_tform(obj.A, [b(1) b(2); b(1)+b(3) b(2); b(1)+b(3) b(2)+b(4); b(1) b(2)+b(4)]);
            box = [min(P(:, 1)), min(P(:, 2)), max(P(:, 1)) - min(P(:, 1)), max(P(:, 2)) - min(P(:, 2))];
        end

        function loseTarget(obj, t)
            obj.Lost      = true;
            obj.LostSince = t;
            obj.Pts       = zeros(0, 2);
            obj.SinceDet  = Inf;
        end
    end

    methods (Static, Access = private)
        function s = emptyTracks()
            s = struct("id", {}, "box", {}, "det", {}, "kf", {}, "misses", {}, "hits", {});
        end
    end
end

function c = box_center(b)
c = [b(:, 1) + b(:, 3) / 2, b(:, 2) + b(:, 4) / 2];
end

function r = expand_box(b, s, frameSize)
c = box_center(b);
r = clip_box([c(1) - s * b(3) / 2, c(2) - s * b(4) / 2, s * b(3), s * b(4)], frameSize, true);
end

function j = pick_near(B, ref, maxDist)
% box in B closest to ref (centre distance < maxDist face widths, similar size); 0 if none
j = 0;
if isempty(B) || isempty(ref) || any(isnan(ref)), return; end
d = vecnorm(box_center(B) - box_center(ref), 2, 2) / ref(3);
ratio = (B(:, 3) .* B(:, 4)) / (ref(3) * ref(4));
d(ratio < 0.5 | ratio > 2) = Inf;
[dmin, k] = min(d);
if dmin < maxDist, j = k; end
end
