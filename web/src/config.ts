// Single sources of truth (README 11.1): patches and parameters come from
// shared/*.json, imported at build time. Nothing here may hard-code them.
import patchesJson from '../../shared/patches.json';
import configJson from '../../shared/config.default.json';

export type GroupName = 'A' | 'B' | 'C';
export interface PatchDef {
  id: number;
  code: string;
  name: string;
  group: GroupName;
  groupIndex: number; // 0=A, 1=B, 2=C (TraceFrame encoding)
  mediapipe: number[];
  canonical: { cx: number; cy: number; w: number; h: number };
}

export type FvConfig = Omit<typeof configJson, 'band_hz' | 'hr_grid_bpm' | 'skin_sanity_cb' | 'skin_sanity_cr'> & {
  band_hz: [number, number];
  hr_grid_bpm: [number, number, number];
  skin_sanity_cb: [number, number];
  skin_sanity_cr: [number, number];
};

export const CONFIG = configJson as FvConfig;

export const PATCHES: PatchDef[] = patchesJson.patches.map((p) => ({
  ...p,
  group: p.group as GroupName,
  groupIndex: (patchesJson.groups as Record<string, number>)[p.group],
}));

export const SKIN_MODEL_CODES: string[] = patchesJson.skin_model_patches;
export const MASKED_RULE_CODES: string[] = patchesJson.masked_rule_patches;
export const patchByCode = (code: string) => PATCHES.find((p) => p.code === code)!;
