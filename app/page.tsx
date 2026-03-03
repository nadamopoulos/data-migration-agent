"use client";

import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import { AlertCircle, AlertTriangle, CheckCircle2, Eye, EyeOff, FileText, Key, Loader2, ShieldAlert, Upload, X } from "lucide-react";
import Papa, { ParseResult } from "papaparse";
import { z } from "zod";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import type { AgentResponse, Mapping, NshotValidationResult as NshotValidationResultType } from "@/lib/schemas";
import { ResponseSchema, NshotValidationResultSchema } from "@/lib/schemas";

type CsvSummary = {
  file: File;
  rowCount: number;
};

type ParsedCsv = {
  headers: string[];
  rows: Record<string, string>[];
};

type Slot = "input" | "target";
type Stage = "upload" | "review" | "result";
type OutputSummary = { rowCount: number; columnCount: number };

const CSV_MIME_TYPES = new Set(["text/csv", "application/vnd.ms-excel"]);
const MAX_AGENT_ATTEMPTS = 3;
const PREVIEW_ROWS = 10;
const API_KEY_STORAGE_KEY = "anthropic-api-key";

const ColumnStatsSchema = z.object({
  column: z.string(),
  exact: z.number(),
  fuzzy: z.number(),
  llm: z.number(),
  structural: z.number(),
  passthrough: z.number(),
  total: z.number()
});

type ColumnStats = z.infer<typeof ColumnStatsSchema>;

const ApplyResponseSchema = z.object({
  csv: z.string(),
  summary: z.object({
    rowCount: z.number(),
    columnCount: z.number()
  }),
  normalisationStats: z.array(ColumnStatsSchema).optional(),
  validation: NshotValidationResultSchema.optional()
});

const isCsvFile = (file: File) =>
  file.name.toLowerCase().endsWith(".csv") || CSV_MIME_TYPES.has(file.type);

const parseCsv = (file: File) =>
  new Promise<ParsedCsv>((resolve, reject) => {
    Papa.parse<Record<string, string>>(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        // Only reject on fatal parse errors (bad quoting, bad delimiter).
        // Ignore non-fatal warnings like TooFewFields / TooManyFields which
        // are common in real-world CSVs with trailing commas or ragged rows.
        const fatalErrors = results.errors.filter(
          (e) => e.type === "Quotes" || e.type === "Delimiter"
        );
        if (fatalErrors.length > 0) {
          reject(new Error(fatalErrors[0].message || "Unable to parse CSV"));
          return;
        }

        const headers = (results.meta.fields ?? []).map((header) => header.trim());
        const rows = normaliseRows(results, headers);
        if (headers.length === 0 || rows.length === 0) {
          reject(new Error("CSV files must include headers and at least one data row."));
          return;
        }

        resolve({ headers, rows });
      },
      error: (error) => reject(error)
    });
  });

const normaliseRows = (
  results: ParseResult<Record<string, string>>,
  headers: string[]
) =>
  results.data.map((row) =>
    Object.fromEntries(headers.map((header) => [header, String(row[header] ?? "").trim()]))
  );

const getSafeErrorMessage = (error: unknown, fallback: string) =>
  error instanceof Error && error.message ? error.message : fallback;

const createPreviewRows = (csvText: string) => {
  const parsed = Papa.parse<Record<string, string>>(csvText, {
    header: true,
    skipEmptyLines: true
  });
  const headers = (parsed.meta.fields ?? []).map((header) => header.trim());
  const rows = normaliseRows(parsed, headers);
  return rows.slice(0, PREVIEW_ROWS);
};

export default function Home() {
  const inputFileRef = useRef<HTMLInputElement>(null);
  const targetFileRef = useRef<HTMLInputElement>(null);

  const [apiKey, setApiKey] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [keyConnected, setKeyConnected] = useState(false);

  const [inputFile, setInputFile] = useState<File | null>(null);
  const [targetFile, setTargetFile] = useState<File | null>(null);
  const [inputParsed, setInputParsed] = useState<ParsedCsv | null>(null);
  const [targetParsed, setTargetParsed] = useState<ParsedCsv | null>(null);
  const [mappingPlan, setMappingPlan] = useState<AgentResponse | null>(null);
  const [editableMappings, setEditableMappings] = useState<Mapping[]>([]);
  const [outputCsv, setOutputCsv] = useState("");
  const [outputSummary, setOutputSummary] = useState<OutputSummary | null>(null);
  const [outputPreview, setOutputPreview] = useState<Record<string, string>[]>([]);
  const [normalisationStats, setNormalisationStats] = useState<ColumnStats[]>([]);
  const [validationResult, setValidationResult] = useState<NshotValidationResultType | null>(null);
  const [stage, setStage] = useState<Stage>("upload");
  const [isLoading, setIsLoading] = useState(false);
  const [loadingAction, setLoadingAction] = useState<"agent" | "apply" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [inputSummary, setInputSummary] = useState<CsvSummary | null>(null);
  const [targetSummary, setTargetSummary] = useState<CsvSummary | null>(null);
  const [draggingSlot, setDraggingSlot] = useState<Slot | null>(null);
  const [inputError, setInputError] = useState("");
  const [targetError, setTargetError] = useState("");

  useEffect(() => {
    const stored = localStorage.getItem(API_KEY_STORAGE_KEY);
    if (stored) {
      setApiKey(stored);
      setKeyConnected(true);
    }
  }, []);

  const handleSaveKey = () => {
    const trimmed = apiKey.trim();
    if (!trimmed) {
      return;
    }
    localStorage.setItem(API_KEY_STORAGE_KEY, trimmed);
    setApiKey(trimmed);
    setKeyConnected(true);
  };

  const handleDisconnectKey = () => {
    localStorage.removeItem(API_KEY_STORAGE_KEY);
    setApiKey("");
    setKeyConnected(false);
    setShowKey(false);
  };

  const resetWorkflowState = () => {
    setMappingPlan(null);
    setEditableMappings([]);
    setOutputCsv("");
    setOutputSummary(null);
    setOutputPreview([]);
    setNormalisationStats([]);
    setValidationResult(null);
    setStage("upload");
  };

  const startOver = () => {
    setInputFile(null);
    setTargetFile(null);
    setInputParsed(null);
    setTargetParsed(null);
    setInputSummary(null);
    setTargetSummary(null);
    setInputError("");
    setTargetError("");
    setError(null);
    resetWorkflowState();
    if (inputFileRef.current) {
      inputFileRef.current.value = "";
    }
    if (targetFileRef.current) {
      targetFileRef.current.value = "";
    }
  };

  const handleChosenFile = async (file: File | null, slot: Slot) => {
    if (!file) {
      return;
    }

    if (!isCsvFile(file)) {
      const errorMessage = "Only CSV files are supported.";
      if (slot === "input") {
        setInputError(errorMessage);
      } else {
        setTargetError(errorMessage);
      }
      return;
    }

    try {
      const parsed = await parseCsv(file);
      const rowCount = parsed.rows.length;
      resetWorkflowState();
      setError(null);

      if (slot === "input") {
        setInputFile(file);
        setInputParsed(parsed);
        setInputSummary({ file, rowCount });
        setInputError("");
      } else {
        setTargetFile(file);
        setTargetParsed(parsed);
        setTargetSummary({ file, rowCount });
        setTargetError("");
      }
    } catch (uploadError) {
      const errorMessage = getSafeErrorMessage(
        uploadError,
        "Could not read this CSV file. Please try another one."
      );
      if (slot === "input") {
        setInputFile(null);
        setInputParsed(null);
        setInputError(errorMessage);
      } else {
        setTargetFile(null);
        setTargetParsed(null);
        setTargetError(errorMessage);
      }
    }
  };

  const onFileInputChange =
    (slot: Slot) =>
    async (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0] ?? null;
      await handleChosenFile(file, slot);
    };

  const onDropFile =
    (slot: Slot) =>
    async (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setDraggingSlot(null);
      const file = event.dataTransfer.files?.[0] ?? null;
      await handleChosenFile(file, slot);
    };

  const openFileDialog = (slot: Slot) => {
    if (slot === "input") {
      inputFileRef.current?.click();
      return;
    }
    targetFileRef.current?.click();
  };

  const runAgentAttempt = async () => {
    if (!inputFile || !targetFile) {
      throw new Error("Please upload both CSV files before running the agent.");
    }

    const payload = new FormData();
    payload.append("inputCsv", inputFile);
    payload.append("targetCsv", targetFile);

    const headers: Record<string, string> = {};
    const storedKey = localStorage.getItem(API_KEY_STORAGE_KEY);
    if (storedKey) {
      headers["x-api-key"] = storedKey;
    }

    const response = await fetch("/api/process", {
      method: "POST",
      headers,
      body: payload
    });

    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(
        typeof body.error === "string"
          ? body.error
          : "The mapping agent request failed."
      );
    }

    const validated = ResponseSchema.safeParse(body);
    if (!validated.success) {
      throw new Error("The mapping plan response was invalid.");
    }

    return validated.data;
  };

  const handleRunAgent = async () => {
    if (!inputParsed || !targetParsed || !inputFile || !targetFile) {
      setError("Upload both CSV files before running the agent.");
      return;
    }

    if (inputParsed.rows.length === 0 || targetParsed.rows.length === 0) {
      setError("CSV files cannot be empty.");
      return;
    }

    if (!keyConnected) {
      setError("Please enter your Anthropic API key first.");
      return;
    }

    setError(null);
    setIsLoading(true);
    setLoadingAction("agent");

    let result: AgentResponse | null = null;
    let lastError = "The mapping agent could not complete your request.";

    for (let attempt = 1; attempt <= MAX_AGENT_ATTEMPTS; attempt += 1) {
      try {
        result = await runAgentAttempt();
        break;
      } catch (runError) {
        lastError = getSafeErrorMessage(
          runError,
          "The mapping agent could not complete your request."
        );
        // Don't retry on client-side validation or auth errors
        const noRetry =
          /api key|empty|both csv|cannot be empty|Unable to generate|could not process/i.test(lastError);
        if (noRetry || attempt === MAX_AGENT_ATTEMPTS) {
          break;
        }
      }
    }

    setIsLoading(false);
    setLoadingAction(null);

    if (!result) {
      setError(lastError);
      return;
    }

    setMappingPlan(result);
    setEditableMappings(result.mappings);
    setStage("review");
  };

  const updateMappingRule = (index: number, nextRule: string) => {
    setEditableMappings((current) =>
      current.map((mapping, mappingIndex) =>
        mappingIndex === index
          ? { ...mapping, normalisationRule: nextRule }
          : mapping
      )
    );
  };

  const handleApproveAndApply = async () => {
    if (!inputParsed || !targetParsed || editableMappings.length === 0) {
      setError("No mapping plan is available to apply.");
      return;
    }

    setError(null);
    setIsLoading(true);
    setLoadingAction("apply");

    try {
      const headers: Record<string, string> = {
        "Content-Type": "application/json"
      };
      const storedKey = localStorage.getItem(API_KEY_STORAGE_KEY);
      if (storedKey) {
        headers["x-api-key"] = storedKey;
      }

      const response = await fetch("/api/apply", {
        method: "POST",
        headers,
        body: JSON.stringify({
          mappings: editableMappings,
          inputCsvData: inputParsed.rows,
          targetCsvData: targetParsed.rows
        })
      });

      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(
          typeof body.error === "string"
            ? body.error
            : "Failed to apply the mapping plan."
        );
      }

      const validated = ApplyResponseSchema.safeParse(body);
      if (!validated.success) {
        throw new Error("Invalid response received while applying the mapping plan.");
      }

      const previewRows = createPreviewRows(validated.data.csv);
      setOutputCsv(validated.data.csv);
      setOutputSummary(validated.data.summary);
      setOutputPreview(previewRows);
      setNormalisationStats(validated.data.normalisationStats ?? []);
      setValidationResult(validated.data.validation ?? null);
      setStage("result");
    } catch (applyError) {
      setError(
        getSafeErrorMessage(
          applyError,
          "Unable to apply mappings to the uploaded CSV data."
        )
      );
    } finally {
      setIsLoading(false);
      setLoadingAction(null);
    }
  };

  const handleDownloadCsv = () => {
    if (!outputCsv) {
      return;
    }

    const blob = new Blob([outputCsv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "normalised-output.csv";
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main className="min-h-screen bg-slate-50 px-4 py-10 md:px-8">
      <div className="mx-auto w-full max-w-6xl space-y-8">
        <header className="space-y-3">
          <h1 className="text-3xl font-semibold tracking-tight text-slate-900 md:text-4xl">
            CSV Normalisation Agent
          </h1>
          <p className="max-w-3xl text-sm text-slate-600 md:text-base">
            Upload your input data and a target format example to automatically
            map and normalise your CSV.
          </p>
        </header>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <Key className="h-4 w-4" />
              Anthropic API Key
            </CardTitle>
          </CardHeader>
          <CardContent>
            {keyConnected ? (
              <div className="flex items-center gap-3">
                <div className="flex-1 rounded-md border bg-slate-50 px-3 py-2 text-sm text-slate-600">
                  Connected: sk-ant-...{apiKey.slice(-4)}
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleDisconnectKey}
                >
                  Disconnect
                </Button>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <div className="relative flex-1">
                  <Input
                    type={showKey ? "text" : "password"}
                    placeholder="sk-ant-api03-..."
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") handleSaveKey();
                    }}
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    className="absolute right-1 top-0 h-9 w-9"
                    onClick={() => setShowKey(!showKey)}
                    type="button"
                    aria-label={showKey ? "Hide key" : "Show key"}
                  >
                    {showKey ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </Button>
                </div>
                <Button onClick={handleSaveKey} disabled={!apiKey.trim()}>
                  Connect
                </Button>
              </div>
            )}
            <p className="mt-2 text-xs text-slate-500">
              Your key is stored in your browser only and sent directly to the Anthropic API.
              Get one at{" "}
              <a
                href="https://console.anthropic.com/settings/keys"
                target="_blank"
                rel="noopener noreferrer"
                className="underline"
              >
                console.anthropic.com
              </a>
            </p>
          </CardContent>
        </Card>

        {error && (
          <Alert className="animate-in fade-in-50 duration-300" variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Something went wrong</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
            <Button
              aria-label="Dismiss error"
              className="absolute right-2 top-2 h-7 w-7 p-0"
              onClick={() => setError(null)}
              size="icon"
              type="button"
              variant="ghost"
            >
              <X className="h-4 w-4" />
            </Button>
          </Alert>
        )}

        <section
          className={cn(
            "grid gap-6 transition-all duration-300 md:grid-cols-2",
            stage !== "upload" && "opacity-90"
          )}
        >
          <Card>
            <CardHeader>
              <CardTitle>Input CSV</CardTitle>
              <CardDescription>Your raw data</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <input
                accept=".csv,text/csv"
                className="hidden"
                onChange={onFileInputChange("input")}
                ref={inputFileRef}
                type="file"
              />
              <div
                className={cn(
                  "flex min-h-40 cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-slate-300 px-4 py-6 text-center transition-colors",
                  draggingSlot === "input" && "border-primary bg-primary/5"
                )}
                onClick={() => openFileDialog("input")}
                onDragEnter={(event) => {
                  event.preventDefault();
                  setDraggingSlot("input");
                }}
                onDragLeave={(event) => {
                  event.preventDefault();
                  setDraggingSlot((current) =>
                    current === "input" ? null : current
                  );
                }}
                onDragOver={(event) => event.preventDefault()}
                onDrop={onDropFile("input")}
              >
                <Upload className="mb-3 h-6 w-6 text-slate-500" />
                <p className="text-sm font-medium text-slate-700">
                  Drag and drop your CSV, or click to browse
                </p>
              </div>
              {inputSummary && (
                <div className="rounded-md bg-slate-100 px-3 py-2 text-sm text-slate-700">
                  <div className="flex items-center gap-2 font-medium">
                    <FileText className="h-4 w-4" />
                    {inputSummary.file.name}
                  </div>
                  <p className="mt-1 text-xs text-slate-600">
                    {inputSummary.rowCount} rows detected
                  </p>
                </div>
              )}
              {inputError && (
                <p className="text-sm font-medium text-destructive">{inputError}</p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Target CSV</CardTitle>
              <CardDescription>Example of desired output format</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <input
                accept=".csv,text/csv"
                className="hidden"
                onChange={onFileInputChange("target")}
                ref={targetFileRef}
                type="file"
              />
              <div
                className={cn(
                  "flex min-h-40 cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-slate-300 px-4 py-6 text-center transition-colors",
                  draggingSlot === "target" && "border-primary bg-primary/5"
                )}
                onClick={() => openFileDialog("target")}
                onDragEnter={(event) => {
                  event.preventDefault();
                  setDraggingSlot("target");
                }}
                onDragLeave={(event) => {
                  event.preventDefault();
                  setDraggingSlot((current) =>
                    current === "target" ? null : current
                  );
                }}
                onDragOver={(event) => event.preventDefault()}
                onDrop={onDropFile("target")}
              >
                <Upload className="mb-3 h-6 w-6 text-slate-500" />
                <p className="text-sm font-medium text-slate-700">
                  Drag and drop your CSV, or click to browse
                </p>
              </div>
              {targetSummary && (
                <div className="rounded-md bg-slate-100 px-3 py-2 text-sm text-slate-700">
                  <div className="flex items-center gap-2 font-medium">
                    <FileText className="h-4 w-4" />
                    {targetSummary.file.name}
                  </div>
                  <p className="mt-1 text-xs text-slate-600">
                    {targetSummary.rowCount} rows detected
                  </p>
                </div>
              )}
              {targetError && (
                <p className="text-sm font-medium text-destructive">{targetError}</p>
              )}
            </CardContent>
          </Card>
        </section>

        {stage === "upload" && (
          <Button
            className="w-full transition-all duration-300"
            disabled={!inputSummary || !targetSummary || !keyConnected || isLoading}
            onClick={handleRunAgent}
          >
            {loadingAction === "agent" ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Agent is thinking...
              </>
            ) : (
              "Run Agent"
            )}
          </Button>
        )}

        {stage === "review" && mappingPlan && (
          <Card className="animate-in fade-in-50 duration-300">
            <CardHeader>
              <CardTitle>Mapping &amp; Normalisation Plan</CardTitle>
              <CardDescription>Edit rules if needed, then approve and apply.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Input Column</TableHead>
                      <TableHead>Target Column</TableHead>
                      <TableHead>Normalisation Rule</TableHead>
                      <TableHead>Type</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {editableMappings.map((mapping, index) => (
                      <TableRow key={`${mapping.inputColumn}-${mapping.targetColumn}-${index}`}>
                        <TableCell className="font-medium">{mapping.inputColumn}</TableCell>
                        <TableCell>{mapping.targetColumn}</TableCell>
                        <TableCell className="min-w-72">
                          <Input
                            onChange={(event) =>
                              updateMappingRule(index, event.target.value)
                            }
                            value={mapping.normalisationRule}
                          />
                        </TableCell>
                        <TableCell>{mapping.transformType}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>

              <p className="text-sm text-slate-600">{mappingPlan.agentNotes}</p>
              {!!mappingPlan.unmappedInputColumns.length && (
                <p className="text-sm text-slate-600">
                  Unmapped input columns: {mappingPlan.unmappedInputColumns.join(", ")}
                </p>
              )}
              {!!mappingPlan.unmappedTargetColumns.length && (
                <p className="text-sm text-slate-600">
                  Unmapped target columns: {mappingPlan.unmappedTargetColumns.join(", ")}
                </p>
              )}

              <div className="flex flex-wrap items-center gap-2 pt-2">
                <Button disabled={isLoading} onClick={handleApproveAndApply}>
                  {loadingAction === "apply" ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Normalising (exact / fuzzy / LLM)...
                    </>
                  ) : (
                    "Approve & Apply"
                  )}
                </Button>
                <Button onClick={startOver} variant="ghost">
                  Start Over
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {stage === "result" && outputSummary && (
          <Card className="animate-in fade-in-50 duration-300">
            <CardHeader>
              <CardTitle>Normalised Output</CardTitle>
              <CardDescription>
                Preview of the first {PREVIEW_ROWS} rows from your transformed CSV.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-slate-700">
                {outputSummary.rowCount} rows • {outputSummary.columnCount} columns
              </p>

              {validationResult && (
                <div className="space-y-3">
                  {validationResult.passed ? (
                    <Alert>
                      <CheckCircle2 className="h-4 w-4 text-green-600" />
                      <AlertTitle className="text-green-800">
                        N-Shot Validation Passed
                      </AlertTitle>
                      <AlertDescription className="text-green-700">
                        {validationResult.sampleSize} sample rows checked — no
                        cross-column contamination or hallucinations detected.
                      </AlertDescription>
                    </Alert>
                  ) : (
                    <Alert variant="destructive">
                      <ShieldAlert className="h-4 w-4" />
                      <AlertTitle>
                        N-Shot Validation Found Issues
                      </AlertTitle>
                      <AlertDescription>
                        {validationResult.violations.length} issue
                        {validationResult.violations.length !== 1 ? "s" : ""}{" "}
                        detected across {validationResult.sampleSize} sample
                        rows. Review the details below before downloading.
                      </AlertDescription>
                    </Alert>
                  )}

                  {/* Per-column confidence */}
                  {validationResult.columnConfidence.length > 0 && (
                    <div className="rounded-md border">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Column</TableHead>
                            <TableHead className="text-center">
                              Confidence
                            </TableHead>
                            <TableHead className="text-center">
                              Violations
                            </TableHead>
                            <TableHead className="text-center">
                              Checked Rows
                            </TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {validationResult.columnConfidence.map((col) => (
                            <TableRow key={`conf-${col.column}`}>
                              <TableCell className="font-medium">
                                {col.column}
                              </TableCell>
                              <TableCell className="text-center">
                                <span
                                  className={cn(
                                    "inline-block rounded-full px-2 py-0.5 text-xs font-medium",
                                    col.confidence >= 0.8
                                      ? "bg-green-100 text-green-800"
                                      : col.confidence >= 0.5
                                        ? "bg-amber-100 text-amber-800"
                                        : "bg-red-100 text-red-800"
                                  )}
                                >
                                  {(col.confidence * 100).toFixed(0)}%
                                </span>
                              </TableCell>
                              <TableCell className="text-center">
                                {col.violations > 0 ? (
                                  <span className="inline-block rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-800">
                                    {col.violations}
                                  </span>
                                ) : (
                                  <span className="text-xs text-slate-400">
                                    0
                                  </span>
                                )}
                              </TableCell>
                              <TableCell className="text-center text-xs text-slate-500">
                                {col.checkedRows}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  )}

                  {/* Violation details */}
                  {validationResult.violations.length > 0 && (
                    <div className="space-y-2">
                      <p className="text-sm font-medium text-slate-800">
                        Violation Details
                      </p>
                      <div className="max-h-64 space-y-2 overflow-y-auto">
                        {validationResult.violations.map((v, idx) => (
                          <div
                            key={`viol-${idx}`}
                            className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm"
                          >
                            <div className="flex items-center gap-2">
                              <AlertTriangle className="h-3.5 w-3.5 flex-shrink-0 text-red-600" />
                              <span className="font-medium text-red-800">
                                {v.type === "cross_column_contamination"
                                  ? "Cross-Column Contamination"
                                  : "Hallucination"}
                              </span>
                              <span className="text-xs text-red-600">
                                Row {v.row + 1} &middot; {v.targetColumn}
                              </span>
                            </div>
                            <p className="mt-1 text-xs text-red-700">
                              {v.detail}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {normalisationStats.length > 0 && (
                <div className="space-y-2">
                  <p className="text-sm font-medium text-slate-800">
                    Normalisation Pipeline Breakdown
                  </p>
                  <div className="rounded-md border">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Column</TableHead>
                          <TableHead className="text-center">Exact Match</TableHead>
                          <TableHead className="text-center">Fuzzy Match</TableHead>
                          <TableHead className="text-center">LLM Reasoning</TableHead>
                          <TableHead className="text-center">Structural</TableHead>
                          <TableHead className="text-center">Total</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {normalisationStats.map((stat) => (
                          <TableRow key={stat.column}>
                            <TableCell className="font-medium">{stat.column}</TableCell>
                            <TableCell className="text-center">
                              {stat.exact > 0 && (
                                <span className="inline-block rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-800">
                                  {stat.exact}
                                </span>
                              )}
                            </TableCell>
                            <TableCell className="text-center">
                              {stat.fuzzy > 0 && (
                                <span className="inline-block rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-800">
                                  {stat.fuzzy}
                                </span>
                              )}
                            </TableCell>
                            <TableCell className="text-center">
                              {stat.llm > 0 && (
                                <span className="inline-block rounded-full bg-purple-100 px-2 py-0.5 text-xs font-medium text-purple-800">
                                  {stat.llm}
                                </span>
                              )}
                            </TableCell>
                            <TableCell className="text-center">
                              {stat.structural > 0 && (
                                <span className="inline-block rounded-full bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-800">
                                  {stat.structural}
                                </span>
                              )}
                            </TableCell>
                            <TableCell className="text-center text-xs text-slate-500">
                              {stat.total}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                  <p className="text-xs text-slate-500">
                    Exact = matched target value directly. Fuzzy = close string match.
                    LLM = Claude reasoned over the value. Structural = format transform (dates, numbers, etc.).
                  </p>
                </div>
              )}

              {outputPreview.length > 0 ? (
                <div className="rounded-md border">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        {Object.keys(outputPreview[0]).map((header) => (
                          <TableHead key={header}>{header}</TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {outputPreview.map((row, rowIndex) => (
                        <TableRow key={`preview-row-${rowIndex}`}>
                          {Object.entries(row).map(([key, value]) => (
                            <TableCell key={`${rowIndex}-${key}`}>{value}</TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              ) : (
                <p className="text-sm text-slate-600">No preview rows available.</p>
              )}

              <div className="flex flex-wrap items-center gap-2">
                <Button onClick={handleDownloadCsv}>Download Normalised CSV</Button>
                <Button onClick={startOver} variant="ghost">
                  Start Over
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </main>
  );
}
