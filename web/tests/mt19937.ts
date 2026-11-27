// MT19937 (Matsumoto & Nishimura, init_genrand + genrand_res53): the same
// stream as MATLAB rng(seed) + rand (default 'twister' generator), so synthetic
// test inputs can be generated bit-identically in MATLAB and TypeScript.
export function matlabRand(seed: number): () => number {
  const mt = new Uint32Array(624);
  let mti = 624;
  mt[0] = seed >>> 0;
  for (let i = 1; i < 624; i++) {
    const s = mt[i - 1] ^ (mt[i - 1] >>> 30);
    mt[i] = (Math.imul(1812433253, s) + i) >>> 0;
  }
  const next = (): number => {
    if (mti >= 624) {
      for (let k = 0; k < 624; k++) {
        const y = (mt[k] & 0x80000000) | (mt[(k + 1) % 624] & 0x7fffffff);
        mt[k] = mt[(k + 397) % 624] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
      }
      mti = 0;
    }
    let y = mt[mti++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  };
  return () => {
    const a = next() >>> 5, b = next() >>> 6;
    return (a * 67108864 + b) / 9007199254740992;
  };
}
