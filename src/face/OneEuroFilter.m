classdef OneEuroFilter < handle
%ONEEUROFILTER  Adaptive low-pass filter for coordinates (Casiez et al., CHI 2012).
%   f = OneEuroFilter(minCutoff, beta, dCutoff)
%   y = f.update(x, t)    x row vector, t time in seconds
%   f.reset()

    properties
        MinCutoff
        Beta
        DCutoff
    end
    properties (Access = private)
        XPrev  = []
        DxPrev = []
        TPrev  = []
    end

    methods
        function obj = OneEuroFilter(minCutoff, beta, dCutoff)
            obj.MinCutoff = minCutoff;
            obj.Beta      = beta;
            obj.DCutoff   = dCutoff;
        end

        function y = update(obj, x, t)
            if isempty(obj.XPrev) || numel(obj.XPrev) ~= numel(x)
                obj.XPrev = x;  obj.DxPrev = zeros(size(x));  obj.TPrev = t;
                y = x;
                return
            end
            dt = t - obj.TPrev;
            if ~(dt > 0), dt = 1/30; end
            dx    = (x - obj.XPrev) / dt;
            aD    = smoothingFactor(obj.DCutoff, dt);
            dxHat = aD * dx + (1 - aD) * obj.DxPrev;
            a     = smoothingFactor(obj.MinCutoff + obj.Beta * abs(dxHat), dt);
            y     = a .* x + (1 - a) .* obj.XPrev;
            obj.XPrev = y;  obj.DxPrev = dxHat;  obj.TPrev = t;
        end

        function reset(obj)
            obj.XPrev = [];  obj.DxPrev = [];  obj.TPrev = [];
        end
    end
end

function a = smoothingFactor(fc, dt)
tau = 1 ./ (2 * pi * fc);
a = 1 ./ (1 + tau / dt);
end
