function Q = apply_tform(A, P)
%APPLY_TFORM  Apply a 3x3 2-D transform A ([x';y';1] = A*[x;y;1]) to points P (Kx2).

Q = (A(1:2, 1:2) * P.' + A(1:2, 3)).';
end
