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
