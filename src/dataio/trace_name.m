function name = trace_name(dataset, subject, condition, constraint)
%TRACE_NAME  Trace file name, e.g. "UBFC2_subject1_Baseline_None.mat".

name = sprintf("%s_%s_%s_%s.mat", dataset, subject, condition, constraint);
end
