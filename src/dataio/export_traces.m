function outPath = export_traces(out, outDir)
%EXPORT_TRACES  Save process_video output as <dataset>_<subject>_<Condition>_<Constraint>.mat.
%   outPath = export_traces(out)            -> outputs/traces/
%   outPath = export_traces(out, outDir)
%   Saved with -v7 and text as char, so scipy.io.loadmat can read it.

arguments
    out struct
    outDir (1,1) string = "outputs/traces"
end

if ~isfolder(outDir), mkdir(outDir); end
m = out.meta;
outPath = fullfile(outDir, trace_name(m.dataset, m.subject, m.condition, m.constraint));
S = to_char_struct(out);
save(outPath, "-struct", "S", "-v7");
end
