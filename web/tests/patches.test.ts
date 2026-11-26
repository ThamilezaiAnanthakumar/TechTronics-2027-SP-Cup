import { describe, expect, it } from 'vitest';
import { PATCHES } from '../src/config';

// Li et al. 2026 Table 2, as quoted in README Section 4.2(b).
const SPEC = [
  'MF  10,109,108,151,337,338',
  'G   151,108,107,55,8,285,336,337',
  'LLF 67,103,104,105,66,107,108,109',
  'RLF 297,338,337,336,296,334,333,332',
  'ND  8,55,193,122,196,3,195,248,419,351,417,285',
  'LM  126,100,118,117,116,123,147,187,205,203,129,209',
  'RM  355,429,358,423,425,411,376,352,345,346,347,329',
  'P   2,97,98,203,186,185,40,39,37,0,267,269,270,409,410,423,327,326',
  'LSM 203,205,187,147,177,215,138,172,136,135,212,186',
  'RSM 423,410,432,364,365,397,367,435,401,376,411,425',
];

describe('shared/patches.json', () => {
  it('has the 10 README 4.2(b) polygons exactly, with groups A/B/C and unique ids', () => {
    expect(PATCHES).toHaveLength(10);
    for (const row of SPEC) {
      const [code, idx] = row.split(/\s+/);
      const p = PATCHES.find((q) => q.code === code)!;
      expect(p.mediapipe).toEqual(idx.split(',').map(Number));
    }
    const groups = Object.fromEntries(PATCHES.map((p) => [p.code, p.group]));
    expect(groups).toEqual({ G: 'A', ND: 'A', LM: 'A', RM: 'A', MF: 'B', LLF: 'B', RLF: 'B', P: 'C', LSM: 'C', RSM: 'C' });
    expect(new Set(PATCHES.map((p) => p.id)).size).toBe(10);
  });
});
