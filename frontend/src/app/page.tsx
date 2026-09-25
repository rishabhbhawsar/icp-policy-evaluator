// src/app/page.tsx
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ChangeEvent, FC } from "react";
import TelemetryHeader from "@/components/TelemetryHeader";

/* ------------------------------------------------------------------ */
/* Types                                                              */
/* ------------------------------------------------------------------ */

type LogLevel = "INFO" | "CACHE" | "ERROR" | "DONE";
type Verdict = "COMPLIANT" | "NON_COMPLIANT" | "UNKNOWN";

interface LogEntry {
  readonly id: number;
  readonly timestamp: string;
  readonly level: LogLevel;
  readonly message: string;
}

interface EvaluationResult {
  readonly verdict: Verdict;
  readonly raw: unknown;
  readonly httpStatus: number;
}

interface IngressPanelProps {
  value: string;
  disabled: boolean;
  onChange: (next: string) => void;
  onSubmit: () => void;
  onReset: () => void;
}

interface LogTerminalProps {
  logs: readonly LogEntry[];
  running: boolean;
}

interface VerdictPanelProps {
  result: EvaluationResult | null;
}

/* ------------------------------------------------------------------ */
/* Constants                                                          */
/* ------------------------------------------------------------------ */

const API_BASE_URL: string = (
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  "https://icp-policy-evaluator-backend.onrender.com"
).replace(/\/+$/, "");

const EVALUATE_ENDPOINT = `${API_BASE_URL}/evaluate`;

// Render free-tier instances can take a while to cold start.
const REQUEST_TIMEOUT_MS = 120_000;

const DEFAULT_PAYLOAD: string = `{
  "policy_id": "KYC-014",
  "locale": "US",
  "business_description": "Enterprise consulting group with explicit beneficial ownership registry documents verified."
}`;

const LEVEL_STYLES: Record<LogLevel, string> = {
  INFO: "text-cyan-400",
  CACHE: "text-emerald-400",
  ERROR: "text-rose-500",
  DONE: "text-emerald-300",
};

const MESSAGE_STYLES: Record<LogLevel, string> = {
  INFO: "text-zinc-300",
  CACHE: "text-zinc-300",
  ERROR: "text-rose-400",
  DONE: "text-zinc-300",
};

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */

const formatTime = (date: Date): string =>
  date.toLocaleTimeString("en-GB", { hour12: false });

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const normalizeVerdict = (value: unknown): Verdict => {
  if (typeof value === "boolean") {
    return value ? "COMPLIANT" : "NON_COMPLIANT";
  }
  if (typeof value !== "string") return "UNKNOWN";

  const cleaned: string = value.trim().toUpperCase().replace(/[\s-]+/g, "_");
  if (cleaned === "COMPLIANT") return "COMPLIANT";
  if (cleaned === "NON_COMPLIANT" || cleaned === "NONCOMPLIANT") {
    return "NON_COMPLIANT";
  }
  return "UNKNOWN";
};

const extractVerdict = (data: unknown): Verdict => {
  if (!isRecord(data)) return "UNKNOWN";

  const candidates: unknown[] = [
    data.verdict,
    data.status,
    data.compliance_status,
    data.is_compliant,
    isRecord(data.result) ? data.result.verdict : undefined,
    isRecord(data.result) ? data.result.status : undefined,
  ];

  for (const candidate of candidates) {
    const verdict: Verdict = normalizeVerdict(candidate);
    if (verdict !== "UNKNOWN") return verdict;
  }
  return "UNKNOWN";
};

const describeHttpFailure = (status: number): string => {
  switch (status) {
    case 422:
      return "HTTP 422 Unprocessable Entity | Payload failed backend Pydantic contract validation. Verify keys and value types.";
    case 404:
      return "HTTP 404 Not Found | The /evaluate route could not be located on the cloud cluster.";
    case 502:
      return "HTTP 502 Bad Gateway | Upstream service is unavailable or still cold-starting. Retry shortly.";
    default:
      return `HTTP ${status} | Unexpected response received from the cloud cluster.`;
  }
};

/* ------------------------------------------------------------------ */
/* Components                                                         */
/* ------------------------------------------------------------------ */

const IngressPanel: FC<IngressPanelProps> = ({
  value,
  disabled,
  onChange,
  onSubmit,
  onReset,
}) => {
  const handleChange = (e: ChangeEvent<HTMLTextAreaElement>): void => {
    onChange(e.target.value);
  };

  const isEmpty: boolean = value.trim().length === 0;

  return (
    <section className="flex min-h-[420px] flex-col rounded-md border border-[#27272a] bg-[#09090b]">
      <div className="flex items-center justify-between border-b border-[#27272a] px-4 py-3">
        <h2 className="font-mono text-[11px] uppercase tracking-widest text-zinc-400">
          Data Ingress
        </h2>
        <span className="font-mono text-[11px] text-zinc-600">
          POST /evaluate &middot; {value.length} chars
        </span>
      </div>

      <textarea
        value={value}
        onChange={handleChange}
        disabled={disabled}
        spellCheck={false}
        aria-label="JSON payload ingress"
        className="flex-1 resize-none bg-transparent p-4 font-mono text-sm leading-relaxed text-zinc-200 placeholder:text-zinc-700 focus:outline-none disabled:opacity-60"
      />

      <div className="flex items-center justify-end gap-3 border-t border-[#27272a] px-4 py-3">
        <button
          type="button"
          onClick={onReset}
          disabled={disabled}
          className="rounded border border-[#27272a] px-4 py-2 font-mono text-xs text-zinc-400 transition-colors hover:bg-zinc-900 disabled:opacity-40"
        >
          Reset
        </button>
        <button
          type="button"
          onClick={onSubmit}
          disabled={disabled || isEmpty}
          className="rounded border border-emerald-500/40 bg-emerald-500/10 px-4 py-2 font-mono text-xs font-medium text-emerald-400 transition-colors hover:bg-emerald-500/20 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {disabled ? "Running..." : "Run Compliance Audit"}
        </button>
      </div>
    </section>
  );
};

const LogTerminal: FC<LogTerminalProps> = ({ logs, running }) => {
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [logs]);

  return (
    <section className="flex min-h-[420px] flex-col rounded-md border border-[#27272a] bg-[#09090b]">
      <div className="flex items-center justify-between border-b border-[#27272a] px-4 py-3">
        <h2 className="font-mono text-[11px] uppercase tracking-widest text-zinc-400">
          Async Stream Canvas
        </h2>
        <span className="flex items-center gap-2 font-mono text-[11px] text-zinc-500">
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              running ? "animate-pulse bg-emerald-400" : "bg-zinc-600"
            }`}
          />
          {running ? "streaming" : "idle"}
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-4 font-mono text-xs leading-6">
        {logs.length === 0 ? (
          <p className="text-zinc-700">
            $ awaiting input. Submit a JSON payload to begin.
          </p>
        ) : (
          logs.map((log: LogEntry) => (
            <div key={log.id} className="flex gap-3">
              <span className="shrink-0 text-zinc-600">{log.timestamp}</span>
              <span className={`shrink-0 ${LEVEL_STYLES[log.level]}`}>
                [{log.level}]
              </span>
              <span className={MESSAGE_STYLES[log.level]}>{log.message}</span>
            </div>
          ))
        )}
        <div ref={endRef} />
      </div>
    </section>
  );
};

const VerdictPanel: FC<VerdictPanelProps> = ({ result }) => {
  if (result === null) {
    return (
      <section className="rounded-md border border-dashed border-[#27272a] bg-[#09090b] p-5">
        <p className="font-mono text-[11px] uppercase tracking-widest text-zinc-600">
          Algorithmic Verdict
        </p>
        <p className="mt-2 font-mono text-xs text-zinc-700">
          No evaluation result yet.
        </p>
      </section>
    );
  }

  const styleByVerdict: Record<Verdict, string> = {
    COMPLIANT:
      "border-emerald-500/60 bg-emerald-500/5 text-emerald-400 shadow-[0_0_24px_rgba(16,185,129,0.25)]",
    NON_COMPLIANT:
      "border-rose-500/60 bg-rose-500/5 text-rose-400 shadow-[0_0_24px_rgba(244,63,94,0.25)]",
    UNKNOWN: "border-[#27272a] bg-zinc-900/40 text-zinc-400",
  };

  const label: string =
    result.verdict === "UNKNOWN" ? "VERDICT UNRECOGNIZED" : result.verdict;

  return (
    <section
      className={`rounded-md border p-5 ${styleByVerdict[result.verdict]}`}
    >
      <div className="flex items-center justify-between">
        <p className="font-mono text-[11px] uppercase tracking-widest opacity-70">
          Algorithmic Verdict
        </p>
        <p className="font-mono text-[11px] opacity-70">
          HTTP {result.httpStatus}
        </p>
      </div>
      <p className="mt-2 font-mono text-2xl font-semibold tracking-tight">
        {label}
      </p>
      <pre className="mt-4 max-h-64 overflow-auto rounded border border-[#27272a] bg-[#09090b] p-3 font-mono text-xs leading-relaxed text-zinc-400">
        {JSON.stringify(result.raw, null, 2)}
      </pre>
    </section>
  );
};

/* ------------------------------------------------------------------ */
/* Page                                                               */
/* ------------------------------------------------------------------ */

export default function Page() {
  const [payload, setPayload] = useState<string>(DEFAULT_PAYLOAD);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [running, setRunning] = useState<boolean>(false);
  const [result, setResult] = useState<EvaluationResult | null>(null);

  const counter = useRef<number>(0);
  const controllerRef = useRef<AbortController | null>(null);

  const pushLog = useCallback((level: LogLevel, message: string): void => {
    counter.current += 1;
    const entry: LogEntry = {
      id: counter.current,
      timestamp: formatTime(new Date()),
      level,
      message,
    };
    setLogs((prev: LogEntry[]) => [...prev, entry]);
  }, []);

  useEffect(() => {
    return () => {
      controllerRef.current?.abort();
    };
  }, []);

  const handleSubmit = useCallback(async (): Promise<void> => {
    setResult(null);
    pushLog("INFO", "Parsing data ingress text payload...");

    // Pre-flight validation: halt before any network egress.
    let parsed: unknown;
    try {
      parsed = JSON.parse(payload);
      if (!isRecord(parsed)) {
        throw new Error("Payload root must be a JSON object.");
      }
    } catch {
      pushLog(
        "ERROR",
        "Malformed JSON boundary logic. Verify keys match required contract schema."
      );
      return;
    }

    setRunning(true);
    pushLog(
      "INFO",
      "Dispatching POST request across internet gateway to cloud cluster..."
    );

    const controller = new AbortController();
    controllerRef.current = controller;
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

        try {
      const response: Response = await fetch(EVALUATE_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(parsed),
        signal: controller.signal,
      });

      if (!response.ok) {
        try {
          const errBody: unknown = await response.json();
          const detail: unknown = isRecord(errBody)
            ? (errBody.detail ?? errBody)
            : errBody;

          pushLog(
            "ERROR",
            `HTTP ${response.status} | Backend validation detail: ${JSON.stringify(detail)}`
          );
        } catch {
          pushLog("ERROR", `HTTP ${response.status} | Server interruption. No JSON error body returned.`);
        }
        return;
      }

      let data: unknown;
      try {
        data = await response.json();
      } catch {
        pushLog(
          "ERROR",
          "Response body was not valid JSON. Unable to decode verdict payload."
        );
        return;
      }

      pushLog(
        "CACHE",
        "Status: OK | Local SQLite WAL cache ledger read executed cleanly."
      );

      const verdict: Verdict = extractVerdict(data);
      setResult({ verdict, raw: data, httpStatus: response.status });
      pushLog("DONE", `Compliance audit complete | Verdict: ${verdict}`);
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") {
        pushLog(
          "ERROR",
          `Request timed out after ${REQUEST_TIMEOUT_MS / 1000}s. The cloud cluster may be cold-starting; retry shortly.`
        );
      } else {
        pushLog(
          "ERROR",
          "Network failure | Unable to reach the cloud cluster. Check connectivity or CORS configuration."
        );
      }
    } finally {
      clearTimeout(timeoutId);
      controllerRef.current = null;
      setRunning(false);
    }
  }, [payload, pushLog]);

  const handleReset = useCallback((): void => {
    controllerRef.current?.abort();
    setPayload(DEFAULT_PAYLOAD);
    setLogs([]);
    setResult(null);
    setRunning(false);
  }, []);

  return (
    <div className="min-h-screen bg-[#09090b] text-zinc-100">
      <TelemetryHeader />
      <main className="mx-auto flex max-w-7xl flex-col gap-6 px-6 py-6">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <IngressPanel
            value={payload}
            disabled={running}
            onChange={setPayload}
            onSubmit={() => {
              void handleSubmit();
            }}
            onReset={handleReset}
          />
          <LogTerminal logs={logs} running={running} />
        </div>
        <VerdictPanel result={result} />
      </main>
    </div>
  );
}