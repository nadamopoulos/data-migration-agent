import { createAnthropic } from "@ai-sdk/anthropic";
import { generateObject } from "ai";
import { z } from "zod";

// ── String similarity (Levenshtein) ──────────────────────────────────

function levenshtein(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;

  let prev = Array.from({ length: n + 1 }, (_, i) => i);
  let curr = new Array<number>(n + 1);

  for (let i = 1; i <= m; i++) {
    curr[0] = i;
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost);
    }
    [prev, curr] = [curr, prev];
  }

  return prev[n];
}

function stringSimilarity(a: string, b: string): number {
  const maxLen = Math.max(a.length, b.length);
  if (maxLen === 0) return 1;
  return 1 - levenshtein(a.toLowerCase(), b.toLowerCase()) / maxLen;
}

// ── Types ────────────────────────────────────────────────────────────

export type MatchStage =
  | "exact"
  | "fuzzy"
  | "llm"
  | "structural"
  | "passthrough";

export type ColumnStats = {
  column: string;
  exact: number;
  fuzzy: number;
  llm: number;
  structural: number;
  passthrough: number;
  total: number;
};

type ResolvedValue = {
  output: string;
  stage: MatchStage;
};

// ── Constants ────────────────────────────────────────────────────────

const CATEGORICAL_MAX_UNIQUE = 200;
const CATEGORICAL_MAX_RATIO = 0.5;
const FUZZY_THRESHOLD = 0.75;
const LLM_BATCH_SIZE = 80;
const STRUCTURAL_TYPES = new Set([
  "date_format",
  "phone_format",
  "number_format",
  "name_split"
]);

// ── Heuristics ───────────────────────────────────────────────────────

function isCategorical(
  uniqueValues: string[],
  totalRows: number
): boolean {
  if (uniqueValues.length === 0 || totalRows === 0) return false;
  if (uniqueValues.length > CATEGORICAL_MAX_UNIQUE) return false;
  return uniqueValues.length / totalRows <= CATEGORICAL_MAX_RATIO;
}

// ── Stage 1: Exact match (case-insensitive) ──────────────────────────

function exactMatch(
  value: string,
  targetLookup: Map<string, string>
): string | null {
  const key = value.trim().toLowerCase();
  if (!key) return null;
  return targetLookup.get(key) ?? null;
}

// ── Stage 2: Fuzzy match ─────────────────────────────────────────────

function fuzzyMatch(
  value: string,
  targetValues: string[]
): string | null {
  const normalized = value.trim().toLowerCase();
  if (!normalized) return null;

  let bestMatch: string | null = null;
  let bestScore = 0;

  for (const target of targetValues) {
    const score = stringSimilarity(normalized, target.toLowerCase());
    if (score > bestScore && score >= FUZZY_THRESHOLD) {
      bestScore = score;
      bestMatch = target;
    }
  }

  return bestMatch;
}

// ── Stage 3: LLM batch normalisation ─────────────────────────────────

const LlmNormalisationSchema = z.object({
  normalizedValues: z.array(
    z.object({
      input: z.string(),
      output: z.string()
    })
  )
});

async function llmBatchNormalise(
  unmatchedValues: string[],
  targetValues: string[],
  columnName: string,
  normalisationRule: string,
  apiKey: string
): Promise<Map<string, string>> {
  if (unmatchedValues.length === 0) return new Map();

  const allResults = new Map<string, string>();

  for (let i = 0; i < unmatchedValues.length; i += LLM_BATCH_SIZE) {
    const batch = unmatchedValues.slice(i, i + LLM_BATCH_SIZE);

    const prompt = `You are a data normalisation expert. Your task is to map input values so they match the format and vocabulary of a target dataset column.

Target column: "${columnName}"
Normalisation context: ${normalisationRule || "Match the target values as closely as possible."}

Known valid target values (reference examples from the target dataset):
${JSON.stringify(targetValues.slice(0, 100), null, 2)}

Input values that need normalisation:
${JSON.stringify(batch, null, 2)}

For each input value determine the correct normalised output. Consider:
- Abbreviations and expansions (e.g. "US" → "United States", "M" → "Male", "NY" → "New York")
- Spelling variations and typos (e.g. "colour" → "color", "Untied States" → "United States")
- Different representations of the same concept (e.g. "NYC" → "New York City", "1st" → "First")
- Format differences (e.g. "john doe" → "John Doe", "01/02" → "January 2")
- Domain-specific mappings based on the column context

IMPORTANT: If an input value clearly maps to one of the known target values use that exact target value. If there is no reasonable match, format the value to match the general style and pattern of the target values.

Return ALL input values with their normalised output.`;

    try {
      const anthropic = createAnthropic({ apiKey });
      const result = await generateObject({
        model: anthropic("claude-sonnet-4-5"),
        schema: LlmNormalisationSchema,
        prompt
      });

      for (const item of result.object.normalizedValues) {
        allResults.set(item.input, item.output);
      }
    } catch {
      // LLM failed for this batch — keep values as-is
      for (const value of batch) {
        allResults.set(value, value);
      }
    }
  }

  return allResults;
}

// ── Main pipeline ────────────────────────────────────────────────────

export type MappingInput = {
  inputColumn: string;
  targetColumn: string;
  normalisationRule: string;
  transformType: string;
};

export type NormalisePipelineOptions = {
  mappings: MappingInput[];
  inputRows: Record<string, string>[];
  targetRows: Record<string, string>[];
  apiKey: string | null;
  structuralTransform: (value: string, mapping: MappingInput) => string;
};

export async function normalisePipeline(
  options: NormalisePipelineOptions
): Promise<{
  rows: Record<string, string>[];
  stats: ColumnStats[];
}> {
  const { mappings, inputRows, targetRows, apiKey, structuralTransform } =
    options;
  const outputRows: Record<string, string>[] = inputRows.map(() => ({}));
  const allStats: ColumnStats[] = [];

  for (const mapping of mappings) {
    const stats: ColumnStats = {
      column: mapping.targetColumn,
      exact: 0,
      fuzzy: 0,
      llm: 0,
      structural: 0,
      passthrough: 0,
      total: inputRows.length
    };

    // Collect unique target values for this column
    const targetValuesSet = new Set<string>();
    for (const row of targetRows) {
      const val = row[mapping.targetColumn];
      if (val != null && String(val).trim() !== "") {
        targetValuesSet.add(String(val).trim());
      }
    }
    const targetValues = Array.from(targetValuesSet);

    // Case-insensitive lookup
    const targetLookup = new Map<string, string>();
    for (const val of targetValues) {
      targetLookup.set(val.toLowerCase(), val);
    }

    const isStructural = STRUCTURAL_TYPES.has(mapping.transformType);
    const categorical = isCategorical(targetValues, targetRows.length);
    const useValueMatching =
      categorical && !isStructural && targetValues.length > 0;

    // ── Structural-only path ──
    if (!useValueMatching) {
      for (let i = 0; i < inputRows.length; i++) {
        const raw = inputRows[i][mapping.inputColumn] ?? "";
        outputRows[i][mapping.targetColumn] = structuralTransform(
          String(raw),
          mapping
        );
        stats.structural++;
      }
      allStats.push(stats);
      continue;
    }

    // ── 3-stage value normalisation ──
    const resolved: (ResolvedValue | null)[] = new Array(
      inputRows.length
    ).fill(null);

    // Stage 1: Exact match (raw value, then structurally-transformed)
    for (let i = 0; i < inputRows.length; i++) {
      const raw = String(inputRows[i][mapping.inputColumn] ?? "").trim();

      if (raw === "") {
        resolved[i] = { output: "", stage: "passthrough" };
        stats.passthrough++;
        continue;
      }

      const directMatch = exactMatch(raw, targetLookup);
      if (directMatch !== null) {
        resolved[i] = { output: directMatch, stage: "exact" };
        stats.exact++;
        continue;
      }

      // Try after structural transform (handles casing, trimming, etc.)
      const transformed = structuralTransform(raw, mapping);
      const transformedMatch = exactMatch(transformed, targetLookup);
      if (transformedMatch !== null) {
        resolved[i] = { output: transformedMatch, stage: "exact" };
        stats.exact++;
      }
    }

    // Stage 2: Fuzzy match for unresolved
    for (let i = 0; i < inputRows.length; i++) {
      if (resolved[i] !== null) continue;

      const raw = String(inputRows[i][mapping.inputColumn] ?? "").trim();
      const match = fuzzyMatch(raw, targetValues);
      if (match !== null) {
        resolved[i] = { output: match, stage: "fuzzy" };
        stats.fuzzy++;
      }
    }

    // Stage 3: LLM reasoning for remaining unresolved
    // Group by unique value so each distinct input is normalised once
    const unresolvedMap = new Map<string, number[]>();
    for (let i = 0; i < inputRows.length; i++) {
      if (resolved[i] !== null) continue;
      const raw = String(inputRows[i][mapping.inputColumn] ?? "").trim();
      const existing = unresolvedMap.get(raw);
      if (existing) {
        existing.push(i);
      } else {
        unresolvedMap.set(raw, [i]);
      }
    }

    if (unresolvedMap.size > 0 && apiKey) {
      const llmMapping = await llmBatchNormalise(
        Array.from(unresolvedMap.keys()),
        targetValues,
        mapping.targetColumn,
        mapping.normalisationRule,
        apiKey
      );

      for (const [inputVal, indices] of unresolvedMap) {
        const normalised = llmMapping.get(inputVal) ?? inputVal;
        for (const idx of indices) {
          resolved[idx] = { output: normalised, stage: "llm" };
          stats.llm++;
        }
      }
    } else if (unresolvedMap.size > 0) {
      // No API key — fall back to structural transform
      for (const [inputVal, indices] of unresolvedMap) {
        const transformed = structuralTransform(inputVal, mapping);
        for (const idx of indices) {
          resolved[idx] = { output: transformed, stage: "structural" };
          stats.structural++;
        }
      }
    }

    // Write to output rows
    for (let i = 0; i < inputRows.length; i++) {
      outputRows[i][mapping.targetColumn] = resolved[i]?.output ?? "";
    }

    allStats.push(stats);
  }

  return { rows: outputRows, stats: allStats };
}
