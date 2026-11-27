function [hr, tc] = ppg_hr(tg, ppg, opts)
%PPG_HR  Reference HR (bpm) from the contact PPG, same windows and method as sanity_hr.
%   [hr, tc] = ppg_hr(gt.t, gt.ppg, Win=10, Step=1, Band=[0.7 3], Centers=[])

arguments
    tg (:,1) double
    ppg (:,1) double
    opts.Win (1,1) double = 10
    opts.Step (1,1) double = 1
    opts.Band (1,2) double = [0.7 3.0]
    opts.Centers double = []
end

fs = 30;
[tg, iu] = unique(tg);
tu = (tg(1) : 1/fs : tg(end)).';
x  = interp1(tg, ppg(iu), tu, "linear");
[b, a] = butter(3, opts.Band / (fs/2), "bandpass");
x = filtfilt(b, a, x);
[hr, tc] = spectral_hr(x, tu, fs, Win=opts.Win, Step=opts.Step, Band=opts.Band, Centers=opts.Centers);
end
