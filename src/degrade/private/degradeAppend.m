function degradeAppend(info, name, value)
%DEGRADEAPPEND Add one measured field to an existing sidecar JSON (private).
s = jsondecode(fileread(info.sidecar));
s.(name) = value;
fid = fopen(info.sidecar, 'w'); fwrite(fid, jsonencode(s, 'PrettyPrint', true)); fclose(fid);
end
