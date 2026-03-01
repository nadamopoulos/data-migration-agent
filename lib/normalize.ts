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

const FUZZY_THRESHOLD = 0.7;
const LLM_BATCH_SIZE = 200;
const LLM_CONCURRENCY = 8;
const LLM_CALL_TIMEOUT_MS = 30_000;
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
  targetValues: string[],
  columnName: string,
  normalisationRule: string
): string {
  return `You are a data normalisation expert migrating messy source data into a clean target schema.

TARGET COLUMN: "${columnName}"
RULE / CONTEXT: ${normalisationRule || "Normalise each value to match the target vocabulary and format."}

REFERENCE — the target dataset uses these values for this column:
${JSON.stringify(targetValues.slice(0, 150), null, 2)}

INPUT VALUES to normalise (each must produce exactly one output):
${JSON.stringify(batch, null, 2)}

NORMALISATION GUIDELINES:
1. If the input value is an obvious match for one of the reference target values, output that EXACT target value (preserve casing, spacing, punctuation).
2. Recognise abbreviations, codes, aliases, and foreign-language labels:
   - Country / region: "US" → "United States", "DE" → "Germany"
   - OS names & codenames: "Jammy" → "Ubuntu 22.04 LTS", "Ws2022" → "Windows Server 2022", "RHEL 9" → "Red Hat Enterprise Linux 9"
   - Language codes: "spa" → the target representation (e.g. "es" or "Spanish"), "spanisch" (German for Spanish) → correct target value, "zh_cn" → "zh-CN"
   - Boolean / status: "yes"/"true"/"1"/"on"/"active" → the target's active value; "no"/"false"/"0"/"off"/"inactive"/"disabled" → the target's inactive value; ambiguous flags like "p"/"i"/"a" → infer from column context
   - Environment: "prod"/"production"/"live"/"p" → target's production value; "dev"/"development"/"d" → target's dev value; "uat"/"staging"/"qa"/"quality assurance" → appropriate target value
   - Severity / priority: "sev1"/"critical"/"p1" → target's highest severity; "sev3"/"minor"/"p4"/"low" → target's lowest
3. Typos and near-misses: fix obvious misspellings ("Untied" → "United", "Winodws" → "Windows").
4. If no reference target value fits, produce a value that matches the STYLE (casing, format pattern) of the reference values.
5. Never leave a value unnormalised — every input must map to a meaningful output.

Return ALL input values.`;
}

async function llmBatchNormalise(
  unmatchedValues: string[],
  targetValues: string[],
  columnName: string,
  normalisationRule: string,
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
            model: anthropic("claude-haiku-4-5"),
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
        } catch {
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
  limiter: Limiter
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
        limiter
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
