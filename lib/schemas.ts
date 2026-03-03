import { z } from "zod";

export const MappingSchema = z.object({
  inputColumn: z.string(),
  targetColumn: z.string(),
  normalisationRule: z
    .string()
    .describe("Plain English description of the transformation to apply"),
  transformType: z.enum([
    "rename",
    "date_format",
    "casing",
    "phone_format",
    "name_split",
    "number_format",
    "trim",
    "custom"
  ])
});

export const ResponseSchema = z.object({
  mappings: z.array(MappingSchema),
  unmappedInputColumns: z.array(z.string()),
  unmappedTargetColumns: z.array(z.string()),
  agentNotes: z.string()
});

export type Mapping = z.infer<typeof MappingSchema>;
export type AgentResponse = z.infer<typeof ResponseSchema>;

// ── N-shot validation schemas ────────────────────────────────────────

export const ValidationViolationSchema = z.object({
  row: z.number(),
  targetColumn: z.string(),
  inputColumn: z.string(),
  inputValue: z.string(),
  outputValue: z.string(),
  type: z.enum(["cross_column_contamination", "hallucination"]),
  detail: z.string()
});

export const ColumnConfidenceSchema = z.object({
  column: z.string(),
  confidence: z.number(),
  violations: z.number(),
  checkedRows: z.number()
});

export const NshotValidationResultSchema = z.object({
  sampleSize: z.number(),
  passed: z.boolean(),
  violations: z.array(ValidationViolationSchema),
  columnConfidence: z.array(ColumnConfidenceSchema)
});

export type ValidationViolation = z.infer<typeof ValidationViolationSchema>;
export type ColumnConfidence = z.infer<typeof ColumnConfidenceSchema>;
export type NshotValidationResult = z.infer<typeof NshotValidationResultSchema>;
