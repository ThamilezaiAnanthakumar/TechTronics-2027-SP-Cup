%PROBE_IMNOISE_POISSON Does imnoise(I,'poisson') on a double image add photon noise? (README 5 note)
% Synthetic constant images only. Result recorded in docs/API_NOTES.md.
I = 0.05 * ones(256);                                  % attenuated grey (12.75 of 255)
Jd = imnoise(I, 'poisson');                            % double input
Ju = imnoise(im2uint8(I), 'poisson');                  % uint8 input (value = photon count)
rng(0); Nphot = 5000; Jp = poissrnd(I * Nphot) / Nphot;% explicit photon budget
fprintf('double : mean %.6f  std %.2e  (expected std if 1e12 scaling: %.2e)\n', mean(Jd(:)), std(Jd(:)), sqrt(I(1)/1e12));
fprintf('uint8  : mean %.3f  std %.3f grey levels (sqrt(13) = %.3f)\n', mean(double(Ju(:))), std(double(Ju(:))), sqrt(13));
fprintf('poissrnd Nphot=%d : mean %.6f  std %.2e  (expected %.2e)\n', Nphot, mean(Jp(:)), std(Jp(:)), sqrt(I(1)/Nphot));
fprintf('MATLAB %s, IPT %s\n', version, ver('images').Version);
