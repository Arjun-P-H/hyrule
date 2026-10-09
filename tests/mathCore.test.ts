import { describe, it, expect } from 'bun:test';
import { MARCH, QUALITY_PRESETS, SHADOW } from '../src/lib/constants';
import { TERRAIN_FS } from '../src/components/terrain/demoShader';

describe('§16/§18/§26 march and shadow constants mirror constants.ts', () => {
  const glsl = new Map<string, number>();
  for (const m of TERRAIN_FS.matchAll(/const\s+(?:float|int)\s+([^;]+);/g)) {
    for (const decl of m[1].split(',')) {
      const [name, expr] = decl.split('=').map((x) => x.trim());
      if (name && expr && /^[\d.eE+-]+$/.test(expr)) glsl.set(name, Number(expr));
    }
  }
  const pairs: Array<[string, number]> = [
    ['LAMBDA', MARCH.lambdaStep], ['DMIN', MARCH.deltaMin], ['GROWTH', MARCH.stepGrowth],
    ['START_T', MARCH.startT], ['BISECT', MARCH.bisectSteps], ['BISECT_TOL', MARCH.bisectTol],
    ['CLAMP_SLOPE', MARCH.clampSlope], ['CLAMP_SLOPE_FAR', MARCH.clampSlopeFar], ['CLAMP_BASE', MARCH.clampBase],
    ['GRAD_GATE', MARCH.gradGate], ['GRAD_EPS_MIN', MARCH.gradEpsMin], ['GRAD_CADENCE', MARCH.gradCadence],
    ['SH_START', SHADOW.startT], ['SH_FOOT', SHADOW.footprintScale], ['SH_K', SHADOW.penumbraK],
    ['SH_CUTOFF', SHADOW.cutoff], ['SH_STEP_SCALE', SHADOW.stepScale], ['SH_STEP_MIN', SHADOW.stepMin],
    ['SH_STEP_MAX', SHADOW.stepMax], ['SH_GROWTH', SHADOW.stepGrowth],
  ];
  it.each(pairs)('%s', (name, value) => {
    expect(glsl.get(name)).toBe(value);
  });

  it('the far clamp only ever widens the near clamp', () => {
    expect(MARCH.clampSlopeFar).toBeGreaterThanOrEqual(MARCH.clampSlope);
  });
});

describe('§26/§18 preset budgets (re-profiled in Phase 7)', () => {
  it('march budgets rise with preset; shadow budgets never fall', () => {
    const [low, med, high] = QUALITY_PRESETS;
    expect([low.marchSteps, med.marchSteps, high.marchSteps]).toEqual([64, 96, 128]);
    expect([low.shadowSteps, med.shadowSteps, high.shadowSteps]).toEqual([20, 28, 28]);
    expect(low.shadowSteps).toBeLessThanOrEqual(med.shadowSteps);
    expect(med.shadowSteps).toBeLessThanOrEqual(high.shadowSteps);
  });
  it('budgets fit the shader loop caps (160 march, 40 shadow)', () => {
    for (const p of QUALITY_PRESETS) {
      expect(p.marchSteps).toBeLessThanOrEqual(160);
      expect(p.shadowSteps).toBeLessThanOrEqual(40);
    }
    expect(TERRAIN_FS).toContain('for (int i = 0; i < 160; i++)');
    expect(TERRAIN_FS).toContain('for (int i = 0; i < 40; i++)');
  });
});
