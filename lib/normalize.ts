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

const FUZZY_THRESHOLD = 0.85;
const LLM_BATCH_SIZE = 100;
const LLM_CONCURRENCY = 5;
const LLM_CALL_TIMEOUT_MS = 90_000;
const STRUCTURAL_TYPES = new Set([
  "date_format",
  "phone_format",
  "number_format",
  "name_split"
]);

// ── Concurrency limiter ──────────────────────────────────────────────

type Limiter = <T>(fn: () => Promise<T>) => Promise<T>;

function createLimiter(concurrency: number): Limiter {
  let active = 0;
  const queue: (() => void)[] = [];

  return async function <T>(fn: () => Promise<T>): Promise<T> {
    if (active >= concurrency) {
      await new Promise<void>((resolve) => queue.push(resolve));
    }
    active++;
    try {
      return await fn();
    } finally {
      active--;
      queue.shift()?.();
    }
  };
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

// ── Column rule inference ────────────────────────────────────────────

const ColumnRulesSchema = z.object({
  rules: z.array(
    z.object({
      column: z.string(),
      formatRule: z.string()
    })
  )
});

function buildRuleInferencePrompt(
  columns: { name: string; examples: string[]; hint: string }[]
): string {
  const columnsBlock = columns
    .map(
      (c) =>
        `- Column "${c.name}":\n  Reference examples: ${JSON.stringify(c.examples)}\n  Mapping hint: "${c.hint}"`
    )
    .join("\n");

  return `You are a data format analyst. For each column below, you are given a few EXAMPLE values from a target system. These examples illustrate the expected format — they are NOT an exhaustive list of allowed values. Infer the precise format rule that governs them.

${columnsBlock}

For each column, produce a formatRule: a specific, unambiguous instruction that describes how to normalise ANY input value into the correct format. The rule must capture:
- The exact format pattern (e.g. "BCP-47 locale codes in xx-XX format")
- The casing convention (e.g. "Title Case", "lowercase", "UPPERCASE")
- The naming convention and structure (e.g. "AWS region codes like us-east-1")
- The value domain if it is a closed enumeration — list ALL valid values explicitly, not just the examples
- How to handle abbreviations, synonyms, codes, and foreign-language labels

Be extremely specific. "Match target format" is NOT a valid rule.

Good example rules:
- "BCP-47 locale codes: two-letter lowercase language, hyphen, two-letter uppercase region (e.g. en-US, fr-FR, ja-JP). Map language names, native scripts, and legacy locale codes to their BCP-47 equivalent."
- "Title Case status from closed set: Active, Inactive, Deprecated, Pending, Under Review. Map boolean-like values (yes/no/true/false/1/0/enabled/disabled) to Active or Inactive."
- "AWS region format: {area}-{direction}-{number} e.g. us-east-1, eu-central-1, ap-southeast-1. Map friendly names, abbreviations, and data-centre codes to the canonical AWS region code."
- "Priority scale P1 (critical/highest) through P5 (lowest/trivial). Map severity labels, numbers, and shorthand to the P1–P5 scale."
- "Title Case full environment name from set: Production, Development, Staging, QA, UAT. Map abbreviations (prod, dev, stg) and variants to the canonical name."`;
}

async function inferColumnRules(
  columns: { name: string; examples: string[]; hint: string }[],
  apiKey: string,
  limiter: Limiter
): Promise<Map<string, string>> {
  if (columns.length === 0) return new Map();

  return limiter(async () => {
    try {
      const anthropic = createAnthropic({ apiKey });
      const controller = new AbortController();
      const timeout = setTimeout(
        () => controller.abort(),
        LLM_CALL_TIMEOUT_MS
      );
      const result = await generateObject({
        model: anthropic("claude-sonnet-4-5"),
        schema: ColumnRulesSchema,
        prompt: buildRuleInferencePrompt(columns),
        abortSignal: controller.signal
      });
      clearTimeout(timeout);

      const rulesMap = new Map<string, string>();
      for (const rule of result.object.rules) {
        rulesMap.set(rule.column, rule.formatRule);
      }
      return rulesMap;
    } catch (err) {
      console.error(
        "[normalise] Column rule inference failed:",
        err instanceof Error ? err.message : err
      );
      return new Map();
    }
  });
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

function buildLlmPrompt(
  batch: string[],
  targetExamples: string[],
  columnName: string,
  formatRule: string,
  normalisationRule: string
): string {
  const rule =
    formatRule ||
    normalisationRule ||
    "Transform each value to match the format shown in the reference examples.";

  return `You are a data normalisation expert. Transform each input value for the "${columnName}" column according to the format rule below.

FORMAT RULE for "${columnName}":
${rule}

REFERENCE EXAMPLES (showing the correct output format — these are NOT the only allowed values):
${JSON.stringify(targetExamples.slice(0, 50), null, 2)}

INPUT VALUES to normalise:
${JSON.stringify(batch, null, 2)}

INSTRUCTIONS:
1. Apply the FORMAT RULE to transform every input value into the correct format. The output must follow the exact same pattern as the reference examples.
2. You are NOT limited to only the reference example values. Any value that follows the format rule is valid output.
3. Resolve abbreviations, codes, aliases, typos, and foreign-language labels to their normalised form following the format rule.
4. If the input is already in the correct format, return it unchanged.
5. Every input MUST produce a meaningful output — never return the input unchanged unless it already matches the format rule exactly.

Return ALL ${batch.length} input values.`;
}

async function llmBatchNormalise(
  unmatchedValues: string[],
  targetValues: string[],
  columnName: string,
  normalisationRule: string,
  formatRule: string,
  apiKey: string,
  limiter: Limiter
): Promise<Map<string, string>> {
  if (unmatchedValues.length === 0) return new Map();

  // Split into batches
  const batches: string[][] = [];
  for (let i = 0; i < unmatchedValues.length; i += LLM_BATCH_SIZE) {
    batches.push(unmatchedValues.slice(i, i + LLM_BATCH_SIZE));
  }

  // Fire all batches in parallel (concurrency-limited)
  const batchResults = await Promise.all(
    batches.map((batch) =>
      limiter(async () => {
        const prompt = buildLlmPrompt(
          batch,
          targetValues,
          columnName,
          formatRule,
          normalisationRule
        );
        try {
          const anthropic = createAnthropic({ apiKey });
          const controller = new AbortController();
          const timeout = setTimeout(
            () => controller.abort(),
            LLM_CALL_TIMEOUT_MS
          );
          const result = await generateObject({
            model: anthropic("claude-sonnet-4-5"),
            schema: LlmNormalisationSchema,
            prompt,
            abortSignal: controller.signal
          });
          clearTimeout(timeout);
          const map = new Map<string, string>();
          for (const item of result.object.normalizedValues) {
            map.set(item.input, item.output);
          }
          return map;
        } catch (err) {
          // Log so failures are visible, then pass through unchanged
          console.error(
            `[normalise] LLM batch failed for "${columnName}" (${batch.length} values):`,
            err instanceof Error ? err.message : err
          );
          const map = new Map<string, string>();
          for (const value of batch) {
            map.set(value, value);
          }
          return map;
        }
      })
    )
  );

  // Merge all batch results
  const allResults = new Map<string, string>();
  for (const batchMap of batchResults) {
    for (const [k, v] of batchMap) {
      allResults.set(k, v);
    }
  }
  return allResults;
}

// ── Per-column processing (stages 1–3) ───────────────────────────────

type ColumnResult = {
  targetColumn: string;
  values: string[];
  stats: ColumnStats;
};

async function processColumn(
  mapping: MappingInput,
  inputRows: Record<string, string>[],
  targetValues: string[],
  targetLookup: Map<string, string>,
  apiKey: string | null,
  structuralTransform: (value: string, mapping: MappingInput) => string,
  limiter: Limiter,
  formatRule: string
): Promise<ColumnResult> {
  const stats: ColumnStats = {
    column: mapping.targetColumn,
    exact: 0,
    fuzzy: 0,
    llm: 0,
    structural: 0,
    passthrough: 0,
    total: inputRows.length
  };

  const isStructural = STRUCTURAL_TYPES.has(mapping.transformType);
  const useValueMatching = !isStructural && targetValues.length > 0;

  // ── Structural-only path (dates, phones, numbers, name splits) ──
  if (!useValueMatching) {
    const values = inputRows.map((row) => {
      const raw = row[mapping.inputColumn] ?? "";
      stats.structural++;
      return structuralTransform(String(raw), mapping);
    });
    return { targetColumn: mapping.targetColumn, values, stats };
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
      formatRule,
      apiKey,
      limiter
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

  const values = resolved.map((r) => r?.output ?? "");
  return { targetColumn: mapping.targetColumn, values, stats };
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

  // Shared concurrency limiter for all LLM calls across all columns
  const limiter = createLimiter(LLM_CONCURRENCY);

  // Pre-compute target values and lookups per column
  const targetData = new Map<
    string,
    { values: string[]; lookup: Map<string, string> }
  >();
  for (const mapping of mappings) {
    if (targetData.has(mapping.targetColumn)) continue;
    const valuesSet = new Set<string>();
    for (const row of targetRows) {
      const val = row[mapping.targetColumn];
      if (val != null && String(val).trim() !== "") {
        valuesSet.add(String(val).trim());
      }
    }
    const values = Array.from(valuesSet);
    const lookup = new Map<string, string>();
    for (const val of values) {
      lookup.set(val.toLowerCase(), val);
    }
    targetData.set(mapping.targetColumn, { values, lookup });
  }

  // Infer format rules from target reference examples
  let columnRules = new Map<string, string>();
  if (apiKey) {
    const seen = new Set<string>();
    const columnsForRuleInference = mappings
      .filter((m) => !STRUCTURAL_TYPES.has(m.transformType))
      .filter((m) => {
        if (seen.has(m.targetColumn)) return false;
        seen.add(m.targetColumn);
        return true;
      })
      .map((m) => ({
        name: m.targetColumn,
        examples: targetData.get(m.targetColumn)?.values ?? [],
        hint: m.normalisationRule
      }))
      .filter((c) => c.examples.length > 0);

    if (columnsForRuleInference.length > 0) {
      columnRules = await inferColumnRules(
        columnsForRuleInference,
        apiKey,
        limiter
      );
    }
  }

  // Process ALL columns in parallel
  const columnResults = await Promise.all(
    mappings.map((mapping) => {
      const { values, lookup } = targetData.get(mapping.targetColumn) ?? {
        values: [],
        lookup: new Map()
      };
      return processColumn(
        mapping,
        inputRows,
        values,
        lookup,
        apiKey,
        structuralTransform,
        limiter,
        columnRules.get(mapping.targetColumn) ?? ""
      );
    })
  );

  // Merge column results into output rows
  const outputRows: Record<string, string>[] = inputRows.map(() => ({}));
  const allStats: ColumnStats[] = [];

  for (const result of columnResults) {
    for (let i = 0; i < inputRows.length; i++) {
      outputRows[i][result.targetColumn] = result.values[i];
    }
    allStats.push(result.stats);
  }

  return { rows: outputRows, stats: allStats };
}
