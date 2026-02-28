import { getServerSession } from "next-auth";
import { cookies } from "next/headers";

import { authOptions } from "@/lib/auth";
import { buildSetCookieHeader, buildDeleteCookieHeader, getStoredApiKey } from "@/lib/api-key";

export async function GET() {
  const session = await getServerSession(authOptions);
  if (!session) {
    return Response.json({ error: "Unauthorized" }, { status: 401 });
  }

  const apiKey = await getStoredApiKey();
  return Response.json({
    hasApiKey: !!apiKey,
    maskedKey: apiKey ? `sk-ant-...${apiKey.slice(-4)}` : null
  });
}

export async function POST(request: Request) {
  const session = await getServerSession(authOptions);
  if (!session) {
    return Response.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await request.json().catch(() => ({}));
  const apiKey = typeof body.apiKey === "string" ? body.apiKey.trim() : "";

  if (!apiKey || !apiKey.startsWith("sk-ant-")) {
    return Response.json(
      { error: "Please enter a valid Anthropic API key (starts with sk-ant-)." },
      { status: 400 }
    );
  }

  const cookie = buildSetCookieHeader(apiKey);
  const cookieStore = await cookies();
  cookieStore.set(cookie.name, cookie.value, cookie.options as Parameters<typeof cookieStore.set>[2]);

  return Response.json({
    hasApiKey: true,
    maskedKey: `sk-ant-...${apiKey.slice(-4)}`
  });
}

export async function DELETE() {
  const session = await getServerSession(authOptions);
  if (!session) {
    return Response.json({ error: "Unauthorized" }, { status: 401 });
  }

  const cookie = buildDeleteCookieHeader();
  const cookieStore = await cookies();
  cookieStore.set(cookie.name, cookie.value, cookie.options as Parameters<typeof cookieStore.set>[2]);

  return Response.json({ hasApiKey: false, maskedKey: null });
}
