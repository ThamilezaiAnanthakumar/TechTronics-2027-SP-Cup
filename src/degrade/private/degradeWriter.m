function [w, outFile] = degradeWriter(outRoot, condition, subject, profile, fps)
%DEGRADEWRITER VideoWriter for <outRoot>/<condition>/<subject>.<ext> (private).
%   profile: "Archival" (lossless Motion JPEG 2000, .mj2; default), "Uncompressed AVI", "MPEG-4".
outDir = fullfile(outRoot, condition);
if ~isfolder(outDir), mkdir(outDir); end
switch string(profile)
    case "Archival",         ext = '.mj2';
    case "Uncompressed AVI", ext = '.avi';
    case "MPEG-4",           ext = '.mp4';
    otherwise, error('degrade:profile', 'Unsupported profile "%s".', profile);
end
outFile = char(fullfile(outDir, [char(subject) ext]));
w = VideoWriter(outFile, char(profile));
w.FrameRate = fps;
end
