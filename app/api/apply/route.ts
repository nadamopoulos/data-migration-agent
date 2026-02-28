import Papa from "papaparse";
import { z } from "zod";

import { MappingSchema } from "@/lib/schemas";

const ApplyRequestSchema = z.object({
  mappings: z.array(MappingSchema).min(1),
  inputCsvData: z.array(z.record(z.string(), z.unknown())).min(1)
});

const twoDigit = (value: number) => value.toString().padStart(2, "0");

const titleCase = (value: string) =>
  value
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => part[0].toUpperCase() + part.slice(1))
    .join(" ");

const parseDateParts = (
  value: string,
  ruleHint: string
): { year: number; month: number; day: number } | null => {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }

  const isoMatch = trimmed.match(/^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$/);
  if (isoMatch) {
    return {
      year: Number(isoMatch[1]),
      month: Number(isoMatch[2]),
      day: Number(isoMatch[3])
    };
  }

  const localeMatch = trimmed.match(/^(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})$/);
  if (localeMatch) {
    const first = Number(localeMatch[1]);
    const second = Number(localeMatch[2]);
    const rawYear = Number(localeMatch[3]);
    const year = rawYear < 100 ? 2000 + rawYear : rawYear;
    const useMonthFirst = /mm\/dd|month.?first|us/i.test(ruleHint);
    const month = useMonthFirst ? first : second;
    const day = useMonthFirst ? second : first;

    return { year, month, day };
  }

  const parsed = new Date(trimmed);
  if (Number.isNaN(parsed.getTime())) {
    return null;
  }

  return {
    year: parsed.getUTCFullYear(),
    month: parsed.getUTCMonth() + 1,
    day: parsed.getUTCDate()
  };
};

const formatDate = (value: string, rule: string) => {
  const parts = parseDateParts(value, rule);
  if (!parts) {
    return value.trim();
  }

  const { year, month, day } = parts;
  if (/dd\/mm\/yyyy/i.test(rule)) {
    return `${twoDigit(day)}/${twoDigit(month)}/${year}`;
  }
  if (/mm\/dd\/yyyy/i.test(rule)) {
    return `${twoDigit(month)}/${twoDigit(day)}/${year}`;
  }
  if (/dd-mm-yyyy/i.test(rule)) {
    return `${twoDigit(day)}-${twoDigit(month)}-${year}`;
  }
  return `${year}-${twoDigit(month)}-${twoDigit(day)}`;
};

const formatPhone = (value: string, rule: string) => {
  const digits = value.replace(/\D/g, "");
  if (!digits) {
    return "";
  }

  if (/e\.?164|\+\d+/i.test(rule)) {
    return `+${digits}`;
  }

  if (digits.length === 10) {
    return `(${digits.slice(0, 3)}) ${digits.slice(3, 6)}-${digits.slice(6)}`;
  }

  if (digits.length === 11 && digits.startsWith("1")) {
    return `+1 (${digits.slice(1, 4)}) ${digits.slice(4, 7)}-${digits.slice(7)}`;
  }

  return digits;
};

const formatNumber = (value: string, rule: string) => {
  const cleaned = value
    .trim()
    .replace(/[^\d.,-]/g, "")
    .replace(/,/g, "");

  const numeric = Number(cleaned);
  if (!Number.isFinite(numeric)) {
    return value.trim();
  }

  const decimalMatch = rule.match(/(\d+)\s*decimal/i);
  const decimals = decimalMatch ? Number(decimalMatch[1]) : undefined;
  const shouldUseGrouping = /thousand|comma|separator/i.test(rule);
  const formatter = new Intl.NumberFormat("en-US", {
    minimumFractionDigits: decimals ?? 0,
    maximumFractionDigits: decimals ?? 6,
    useGrouping: shouldUseGrouping
  });
  const formatted = formatter.format(numeric);
  const currencySymbol = rule.match(/[$£€]/)?.[0];

  return currencySymbol ? `${currencySymbol}${formatted}` : formatted;
};

const splitName = (value: string) => {
  const trimmed = value.trim();
  if (!trimmed) {
    return { firstName: "", lastName: "" };
  }

  if (trimmed.includes(",")) {
    const [lastName, firstName] = trimmed.split(",").map((part) => part.trim());
    return { firstName: firstName ?? "", lastName: lastName ?? "" };
  }

  const parts = trimmed.split(/\s+/).filter(Boolean);
  if (parts.length === 1) {
    return { firstName: parts[0], lastName: "" };
  }

  return {
    firstName: parts[0],
    lastName: parts.slice(1).join(" ")
  };
};

const applyCustomRule = (value: string, rule: string, targetColumn: string) => {
  const hint = `${rule} ${targetColumn}`;
  if (/trim/i.test(hint)) {
    return value.trim();
  }
  if (/upper/i.test(hint)) {
    return value.toUpperCase();
  }
  if (/lower/i.test(hint)) {
    return value.toLowerCase();
  }
  if (/title/i.test(hint)) {
    return titleCase(value);
  }
  if (/date/i.test(hint)) {
    return formatDate(value, rule);
  }
  if (/phone|tel/i.test(hint)) {
    return formatPhone(value, rule);
  }
  if (/number|currency|decimal|amount|price/i.test(hint)) {
    return formatNumber(value, rule);
  }
  return value.trim();
};

const transformValue = ({
  value,
  transformType,
  rule,
  targetColumn
}: {
  value: string;
  transformType: z.infer<typeof MappingSchema>["transformType"];
  rule: string;
  targetColumn: string;
}) => {
  switch (transformType) {
    case "rename":
      return value;
    case "date_format":
      return formatDate(value, rule);
    case "casing":
      if (/upper/i.test(rule)) {
        return value.toUpperCase();
      }
      if (/lower/i.test(rule)) {
        return value.toLowerCase();
      }
      return titleCase(value);
    case "phone_format":
      return formatPhone(value, rule);
    case "name_split": {
      const split = splitName(value);
      if (/first/i.test(targetColumn)) {
        return split.firstName;
      }
      if (/last|surname|family/i.test(targetColumn)) {
        return split.lastName;
      }
      return `${split.firstName} ${split.lastName}`.trim();
    }
    case "number_format":
      return formatNumber(value, rule);
    case "trim":
      return value.trim();
    case "custom":
      return applyCustomRule(value, rule, targetColumn);
    default:
      return value;
  }
};

export async function POST(request: Request) {
  try {
    const payload = await request.json();
    const parsedBody = ApplyRequestSchema.safeParse(payload);

    if (!parsedBody.success) {
      return Response.json(
        { error: "Invalid request body for apply endpoint." },
        { status: 400 }
      );
    }

    const { mappings, inputCsvData } = parsedBody.data;

    const targetColumns = Array.from(
      new Set(mappings.map((mapping) => mapping.targetColumn))
    );

    const transformedRows = inputCsvData.map((row) => {
      const outputRow: Record<string, string> = {};

      for (const mapping of mappings) {
        const sourceValue = row[mapping.inputColumn];
        const value = sourceValue == null ? "" : String(sourceValue);
        outputRow[mapping.targetColumn] = transformValue({
          value,
          transformType: mapping.transformType,
          rule: mapping.normalisationRule,
          targetColumn: mapping.targetColumn
        });
      }

      return outputRow;
    });

    const csv = Papa.unparse({
      fields: targetColumns,
      data: transformedRows
    });

    return Response.json({
      csv,
      summary: {
        rowCount: transformedRows.length,
        columnCount: targetColumns.length
      }
    });
  } catch {
    return Response.json(
      { error: "Unable to apply mapping plan to the CSV data." },
      { status: 500 }
    );
  }
}
