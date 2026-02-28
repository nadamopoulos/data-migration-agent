# CSV Normalisation Agent

A Next.js application that uses AI to automatically map and normalise CSV files. Upload your input data and a target format example, and the agent will generate a transformation plan to convert your data.

## Getting Started

```bash
npm install
npm run dev
```

Set your `ANTHROPIC_API_KEY` in `.env.local`:

```
ANTHROPIC_API_KEY=your-key-here
```

Open [http://localhost:3000](http://localhost:3000) to use the app.

## How It Works

1. **Upload** — Drag and drop your input CSV and a target CSV showing the desired format
2. **Review** — The AI agent analyses both files and proposes column mappings with normalisation rules
3. **Apply** — Edit rules if needed, then approve to transform your data and download the result