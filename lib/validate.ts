import type { MappingInput } from "./normalize";
import { stringSimilarity } from "./normalize";

// ── Constants ────────────────────────────────────────────────────────

/** Default number of sample rows for n-shot validation */
export const DEFAULT_N_SHOT_SIZE = 5;

/** Output closely resembles a value in a *different* input column */
const CONTAMINATION_THRESHOLD = 0.85;

/**
 * Minimum similarity between input → output to consider them "related".
 * Below this, and absent from target vocabulary, the value is suspect.
 */
const RELATEDNESS_THRESHOLD = 0.5;

/** Structural transforms follow deterministic rules — skip value checks */
const STRUCTURAL_TYPES = new Set([
  "rename",
  "trim",
  "date_format",
  "phone_format",
  "number_format",
  "name_split"
]);

// ── Types ────────────────────────────────────────────────────────────

export type ViolationType = "cross_column_contamination" | "hallucination";

export type ValidationViolation = {
  row: number;
  targetColumn: string;
  inputColumn: string;
  inputValue: string;
  outputValue: string;
  type: ViolationType;
  detail: string;
};

export type ColumnConfidence = {
  column: string;
  confidence: number;
  violations: number;
  checkedRows: number;
};

export type NshotValidationResult = {
  sampleSize: number;
  passed: boolean;
  violations: ValidationViolation[];
  columnConfidence: ColumnConfidence[];
};

// ── Core validation ──────────────────────────────────────────────────

/**
 * Validate pipeline output for cross-column contamination and hallucinations.
 *
 * For each of the n sampled rows, every non-structural mapping is checked:
 *
 * 1. **Cross-column contamination** — the output value is more similar to a
 *    value in a *different* input column than to the value in the mapped
 *    input column. This catches the scenario where the LLM grabs data from
 *    the wrong field.
 *
 * 2. **Hallucination** — the output value bears little resemblance to the
 *    input value AND is not present (or close) in the target vocabulary.
 *    This catches fabricated values.
 */
export function nShotValidate(
  sampleInputRows: Record<string, string>[],
  sampleOutputRows: Record<string, string>[],
  mappings: MappingInput[],
  targetVocabulary: Map<string, Set<string>>
): NshotValidationResult {
  const violations: ValidationViolation[] = [];
  const rowCount = Math.min(sampleInputRows.length, sampleOutputRows.length);

  for (let rowIdx = 0; rowIdx < rowCount; rowIdx++) {
    const inputRow = sampleInputRows[rowIdx];
    const outputRow = sampleOutputRows[rowIdx];

    for (const mapping of mappings) {
      // Structural transforms are deterministic — no LLM involvement
      if (STRUCTURAL_TYPES.has(mapping.transformType)) continue;

      const inputValue = (inputRow[mapping.inputColumn] ?? "").trim();
      const outputValue = (outputRow[mapping.targetColumn] ?? "").trim();

      if (!inputValue || !outputValue) continue;

      const inputToOutputSim = stringSimilarity(outputValue, inputValue);

      // ── Check 1: Cross-column contamination ───────────────────────
      // Does the output resemble a value from a DIFFERENT input column
      // more than it resembles the mapped input column's value?

      for (const [otherCol, rawOtherVal] of Object.entries(inputRow)) {
        if (otherCol === mapping.inputColumn) continue;
        const otherVal = (rawOtherVal ?? "").trim();
        if (!otherVal) continue;

        const crossSim = stringSimilarity(outputValue, otherVal);
        if (
          crossSim >= CONTAMINATION_THRESHOLD &&
          crossSim > inputToOutputSim + 0.1
        ) {
          violations.push({
            row: rowIdx,
            targetColumn: mapping.targetColumn,
            inputColumn: mapping.inputColumn,
            inputValue,
            outputValue,
            type: "cross_column_contamination",
            detail: `Output "${outputValue}" resembles "${otherVal}" from column "${otherCol}" (${(crossSim * 100).toFixed(0)}% match) more than the mapped input "${inputValue}" (${(inputToOutputSim * 100).toFixed(0)}% match).`
          });
          break; // One contamination flag per cell is sufficient
        }
      }

      // ── Check 2: Hallucination ────────────────────────────────────
      // Output is not recognisable from the input AND not in the known
      // target vocabulary — likely fabricated.

      const knownTargets = targetVocabulary.get(mapping.targetColumn);
      const isInTargetVocab =
        knownTargets != null &&
        (knownTargets.has(outputValue) ||
          knownTargets.has(outputValue.toLowerCase()));

      if (!isInTargetVocab && inputToOutputSim < RELATEDNESS_THRESHOLD) {
        // Also check similarity to all known target values — the output
        // may be a valid normalisation we haven't seen yet.
        let bestTargetSim = 0;
        if (knownTargets) {
          for (const tv of knownTargets) {
            const sim = stringSimilarity(outputValue, tv);
            if (sim > bestTargetSim) bestTargetSim = sim;
          }
        }

        if (bestTargetSim < CONTAMINATION_THRESHOLD) {
          violations.push({
            row: rowIdx,
            targetColumn: mapping.targetColumn,
            inputColumn: mapping.inputColumn,
            inputValue,
            outputValue,
            type: "hallucination",
            detail: `Output "${outputValue}" has low similarity to input "${inputValue}" (${(inputToOutputSim * 100).toFixed(0)}%) and is not a known target value for "${mapping.targetColumn}".`
          });
        }
      }
    }
  }

  // ── Per-column confidence ────────────────────────────────────────

  const seen = new Set<string>();
  const columnConfidence: ColumnConfidence[] = [];

  for (const mapping of mappings) {
    if (seen.has(mapping.targetColumn)) continue;
    seen.add(mapping.targetColumn);

    const isStructural = STRUCTURAL_TYPES.has(mapping.transformType);
    const checkedRows = isStructural ? 0 : rowCount;

    const colViolations = violations.filter(
      (v) => v.targetColumn === mapping.targetColumn
    );

    const confidence =
      checkedRows === 0
        ? 1
        : Math.max(0, 1 - colViolations.length / checkedRows);

    columnConfidence.push({
      column: mapping.targetColumn,
      confidence,
      violations: colViolations.length,
      checkedRows
    });
  }

  return {
    sampleSize: rowCount,
    passed: violations.length === 0,
    violations,
    columnConfidence
  };
}

// ── Sampling ────────────────────────────────────────────────────────

/**
 * Select n rows spread evenly across the dataset (stratified sampling).
 * Returns the selected rows and their original indices.
 */
export function selectSampleRows<T>(
  rows: T[],
  n: number
): { rows: T[]; indices: number[] } {
  if (rows.length <= n) {
    return { rows: [...rows], indices: rows.map((_, i) => i) };
  }

  const step = rows.length / n;
  const indices: number[] = [];
  for (let i = 0; i < n; i++) {
    indices.push(Math.floor(i * step));
  }

  return {
    rows: indices.map((i) => rows[i]),
    indices
  };
}
