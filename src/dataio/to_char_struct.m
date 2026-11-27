function x = to_char_struct(x)
%TO_CHAR_STRUCT  Recursively convert strings to char / cellstr (for scipy-readable .mat files).

if isstring(x)
    if isscalar(x)
        x = char(x);
    else
        x = cellstr(x);
    end
elseif isstruct(x)
    f = fieldnames(x);
    for i = 1:numel(x)
        for j = 1:numel(f)
            x(i).(f{j}) = to_char_struct(x(i).(f{j}));
        end
    end
elseif iscell(x)
    for i = 1:numel(x)
        x{i} = to_char_struct(x{i});
    end
end
end
