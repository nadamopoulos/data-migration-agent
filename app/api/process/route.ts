import { createAnthropic } from "@ai-sdk/anthropic";
import { generateObject } from "ai";
import Papa, { ParseResult } from "papaparse";

import { ResponseSchema } from "@/lib/schemas";

export const maxDuration = 300;

const MAX_SAMPLE_ROWS = 10;
const MAX_GENERATION_ATTEMPTS = 3;

type ParsedCsv = {
  headers: string[];
  rows: Record<string, string>[];
};

const parseCsvText = (csvText: string): ParsedCsv => {
  const result: ParseResult<Record<string, string>> = Papa.parse(csvText, {
    header: true,
    skipEmptyLines: true
  });

  // Only throw on fatal errors (MissingQuotes, etc.)
  // Ignore non-fatal warnings like TooFewFields / TooManyFields which are
  // common in real-world CSVs and don't prevent usable parsing.
  const fatalErrors = result.errors.filter(
    (e) => e.type === "Quotes" || e.type === "Delimiter"
  );
  if (fatalErrors.length > 0) {
    throw new Error(`CSV parsing failed: ${fatalErrors[0].message}`);
  }

  const headers = (result.meta.fields ?? []).map((header) => header.trim());
  const normalisedRows = result.data.map((row) =>
    Object.fromEntries(
      headers.map((header) => [header, String(row[header] ?? "").trim()])
    )
  );

  return {
    headers,
    rows: normalisedRows
  };
};

const buildPrompt = ({
  inputCsv,
  targetCsv
}: {
  inputCsv: ParsedCsv;
  targetCsv: ParsedCsv;
}) => {
  const inputSample = inputCsv.rows.slice(0, MAX_SAMPLE_ROWS);
  const targetSample = targetCsv.rows.slice(0, MAX_SAMPLE_ROWS);

  return `
You are a CSV normalisation planning agent.

Compare the input CSV and target CSV headers plus sample rows. Produce a robust mapping plan that explains how to transform input data so it matches the target format.

Input CSV headers:
${JSON.stringify(inputCsv.headers, null, 2)}

Input CSV sample rows:
${JSON.stringify(inputSample, null, 2)}

Target CSV headers:
${JSON.stringify(targetCsv.headers, null, 2)}

Target CSV sample rows:
${JSON.stringify(targetSample, null, 2)}

Rules:
1) Identify which input column maps to which target column, even when names differ but meaning is equivalent.
2) Infer normalisation rules from target sample values. Consider:
   - Date formats (DD/MM/YYYY, YYYY-MM-DD, etc.)
   - Text casing (UPPERCASE, lowercase, Title Case)
   - Phone formatting (digits-only source to formatted target)
   - Name splitting (full name to first/last names)
   - Number/currency formatting (symbols, decimals, separators)
   - Whitespace trimming
3) Keep normalisationRule plain English and action-oriented.
4) Use transformType values exactly from this list:
   rename, date_format, casing, phone_format, name_split, number_format, trim, custom
5) Return only data that satisfies the provided schema.
`;
};

const isCsvUpload = (file: File) =>
  file.name.toLowerCase().endsWith(".csv") ||
  file.type === "text/csv" ||
  file.type === "application/vnd.ms-excel";

export async function POST(request: Request) {
  const headerKey = request.headers.get("x-api-key")?.trim();
  const apiKey = headerKey || process.env.ANTHROPIC_API_KEY;

  if (!apiKey) {
    return Response.json(
      { error: "Please enter your Anthropic API key to use the agent." },
      { status: 400 }
    );
  }

  try {
    const formData = await request.formData();
    const inputCsv = formData.get("inputCsv");
    const targetCsv = formData.get("targetCsv");

    if (!(inputCsv instanceof File) || !(targetCsv instanceof File)) {
      return Response.json(
        { error: "Both inputCsv and targetCsv files are required." },
        { status: 400 }
      );
    }

    if (!isCsvUpload(inputCsv) || !isCsvUpload(targetCsv)) {
      return Response.json(
        { error: "Only CSV files are supported." },
        { status: 400 }
      );
    }

    const [inputText, targetText] = await Promise.all([
      inputCsv.text(),
      targetCsv.text()
    ]);

    if (!inputText.trim() || !targetText.trim()) {
      return Response.json(
        { error: "Input CSV and target CSV must not be empty." },
        { status: 400 }
      );
    }

    let parsedInput: ParsedCsv;
    let parsedTarget: ParsedCsv;

    try {
      parsedInput = parseCsvText(inputText);
      parsedTarget = parseCsvText(targetText);
    } catch {
      return Response.json(
        { error: "One or both CSV files could not be parsed." },
        { status: 400 }
      );
    }

    if (parsedInput.headers.length === 0 || parsedTarget.headers.length === 0) {
      return Response.json(
        { error: "Both CSV files must include a header row." },
        { status: 400 }
      );
    }

    if (parsedInput.rows.length === 0 || parsedTarget.rows.length === 0) {
      return Response.json(
        { error: "Both CSV files must include at least one data row." },
        { status: 400 }
      );
    }

    const prompt = buildPrompt({ inputCsv: parsedInput, targetCsv: parsedTarget });
    const anthropic = createAnthropic({ apiKey });

    let lastError = "";
    for (let attempt = 1; attempt <= MAX_GENERATION_ATTEMPTS; attempt += 1) {
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 75_000);

        const result = await generateObject({
          model: anthropic("claude-sonnet-4-6"),
          schema: ResponseSchema,
          prompt,
          abortSignal: controller.signal
        });

        clearTimeout(timeout);

        const validated = ResponseSchema.safeParse(result.object);
        if (!validated.success) {
          throw new Error("Agent response failed schema validation.");
        }

        return Response.json(validated.data);
      } catch (err) {
        lastError =
          err instanceof Error ? err.message : "Unknown generation error";
      }
    }

    return Response.json(
      {
        error: `Unable to generate a valid mapping plan after ${MAX_GENERATION_ATTEMPTS} attempts: ${lastError}`
      },
      { status: 502 }
    );
  } catch (err) {
    const message =
      err instanceof Error ? err.message : "Unknown error";
    return Response.json(
      { error: `The mapping agent could not process your request: ${message}` },
      { status: 500 }
    );
  }
}
