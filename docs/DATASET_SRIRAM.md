# Sri Ramachandra dataset (organiser)

Status: **no ground truth.** GT-free analyses only (README Section 6.1). Never used for tuning or accuracy claims.

## Privacy rules (binding)
- **The subject in the sample appears to be a minor.** Never put Sri Ramachandra faces (frames, crops, overlays, thumbnails) in figures, the report, demos or any committed file.
- The data stays out of git: `.gitignore` excludes `*S_N*`-style video names, `data/**` and all video formats; derived per-frame traces of this data go under `outputs/` (git-ignored) only.
- Organiser questions (README Section 1.5, Q7–Q9): patient-monitor HR/SpO2 per subject and task; whether faces may be shown at all; whether the agreement covers minors.

## Files on this machine (26 Nov 2026)
| Location | Content |
|---|---|
| `D:\SPCup Data\dataset\Sri Ramachandra\` | **empty folder** |
| `C:\Users\mufte\Downloads\S_N103_Resting.mp4` | the only sample on disk |

The full set has not been located; ask the operator / organiser.

## Sample `S_N103_Resting.mp4`
Reported by the operator on 26 Nov 2026. **Not yet verified by a repo script**; `fv.io.loadSriRamachandra` and a probe (`tools/probe_sriram.m`, to be written) must confirm each value and record the git hash here.

| Property | Operator report | Verified by script |
|---|---|---|
| Resolution | 848×480 | – |
| Frame rate | ~29.6 fps, steady timestamps | – |
| Video codec / bitrate | H.264, ~1.4 Mbps | – |
| Duration | ~32 s | – |
| Audio | mono AAC | – |
| Ground truth | none: no sidecar or metadata readings, no monitor/oximeter in frame, no periodic beeps in the audio (spectrogram) | – |
| Scene | outdoors, backlit, face underexposed → real-world C1 evidence | – |
| Quick fixed-box POS | forehead / L cheek / R cheek medians ~104 / 89 / 64 bpm; region-to-region MAE 24–35 bpm; SNR ≈ −4 dB | – |

Interpretation: plain POS is unreliable on this video. Use it (and the rest of the set) as a stress test for the confidence / status logic (README Section 4.8), not for accuracy.

## Naming and protocol
File names match the organiser group's protocol (Arulselvi et al., DELCON 2023): tasks **Resting, Walking, Stairs, Squats**, recorded with patient-monitor HR/SpO2 readings. `S_N103` is presumably subject N103. The monitor readings are therefore expected to exist; README Section 1.5 Q7 asks for them per subject and task.

## Routes to a reference (README Section 6.1, Step 1)
| Route | S_N103_Resting | Other files |
|---|---|---|
| Sidecar / metadata | none (operator) | to check |
| Audio beeps | none (operator, spectrogram) | to check quickly |
| Monitor in frame (OCR) | none (operator) | to check quickly |
| Organiser | question drafted (Q7) | – |
