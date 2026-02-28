import type { Metadata } from "next";

import AuthProvider from "@/components/auth-provider";

import "./globals.css";

export const metadata: Metadata = {
  title: "CSV Normalisation Agent",
  description: "Map and normalise CSV files using AI guidance."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="font-sans antialiased">
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
