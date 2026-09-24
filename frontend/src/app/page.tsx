// src/app/page.tsx
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ChangeEvent, FC } from "react";
import TelemetryHeader from "@/components/TelemetryHeader";

type LogLevel = "INFO" | "CACHE" | "JUDGE" | "WARN" | "DONE";

interface LogEntry {
  id: number;
  timestamp: string;
  level: LogLevel;
  message: string;
}

interface LogStep {
  level: LogLevel;
  message: string;
  delayMs: number;
}

interface IngressPanelProps {
  value: string;
  disabled: boolean;
  onChange: (next: string) => void;
  onSubmit: () => void;
  onReset: () => void;
}

interface LogTerminalProps {
  logs: LogEntry[];
  running: boolean;
}

const LEVEL_STYLES: Record<LogLevel, string> = {
  INFO: "text-cyan-400",
  CACHE: "text-emerald-400",
  JUDGE: "text-violet-400",
  WARN: "text-amber-400",
  DONE: "text-emerald-300",
};

const SIMULATED_PIPELINE: ReadonlyArray<LogStep> = [
  { level: "INFO", message: "Request received. Queuing async evaluation task", delayMs: 300 },
  { level: "INFO", message: "Pydantic Data Contract Verified", delayMs: 500 },
  { level: "CACHE", message: "Content Hash Match Found - SQLite WAL Read executed in 0.4ms", delayMs: 600 },
  { level: "INFO", message: "Dispatching concurrent policy checks via asyncio.gather", delayMs: 700 },
  { level: "JUDGE", message: "Taxonomy policy batch evaluated. Structured JSON schema adhered", delayMs: 900 },
  { level: "JUDGE", message: "Confidence scores computed for all policy verdicts", delayMs: 600 },
  { level: "CACHE", message: "Verdict ledger persisted to SQLite (WAL mode)", delayMs: 500 },
  { level: "DONE", message: "Compliance audit complete", delayMs: 400 },
];

const PLACEHOLDER_PROFILE =
  "Paste a business profile here...\n\nExample:\nBusiness: Acme Wellness Ltd\nLocale: en-IN\nCategory: Dietary supplements\nClaims: \"Clinically proven to boost immunity\"";

const formatTime = (date: Date): string =>
  date.toLocaleTimeString("en-GB", { hour12: false });

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
    <section className="flex min-h-[520px] flex-col rounded-md border border-[#27272a] bg-[#09090b]">
      <div className="flex items-center justify-between border-b border-[#27272a] px-4 py-3">
        <h2 className="font-mono text-[11px] uppercase tracking-widest text-zinc-400">
          Data Ingress
        </h2>
        <span className="font-mono text-[11px] text-zinc-600">
          {value.length} chars
        </span>
      </div>

      <textarea
        value={value}
        onChange={handleChange}
        disabled={disabled}
        spellCheck={false}
        placeholder={PLACEHOLDER_PROFILE}
        className="flex-1 resize-none bg-transparent p-4 font-mono text-sm leading-relaxed text-zinc-200 placeholder:text-zinc-700 focus:outline-none disabled:opacity-60"
      />

      <div className="flex items-center justify-end gap-3 border-t border-[#27272a] px-4 py-3">
        <button
          type="button"
          onClick={onReset}
          disabled={disabled}
          className="rounded border border-[#27272a] px-4 py-2 font-mono text-xs text-zinc-400 transition-colors hover:bg-zinc-900 disabled:opacity-40"
        >
          Clear
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
    <section className="flex min-h-[520px] flex-col rounded-md border border-[#27272a] bg-[#09090b]">
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
            $ awaiting input. Submit a business profile to begin.
          </p>
        ) : (
          logs.map((log: LogEntry) => (
            <div key={log.id} className="flex gap-3">
              <span className="shrink-0 text-zinc-600">{log.timestamp}</span>
              <span className={`shrink-0 ${LEVEL_STYLES[log.level]}`}>
                [{log.level}]
              </span>
              <span className="text-zinc-300">{log.message}</span>
            </div>
          ))
        )}
        <div ref={endRef} />
      </div>
    </section>
  );
};

export default function Page() {
  const [profile, setProfile] = useState<string>("");
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [running, setRunning] = useState<boolean>(false);
  const timeouts = useRef<ReturnType<typeof setTimeout>[]>([]);
  const counter = useRef<number>(0);

  const clearTimers = useCallback((): void => {
    timeouts.current.forEach((t) => clearTimeout(t));
    timeouts.current = [];
  }, []);

  useEffect(() => clearTimers, [clearTimers]);

  const handleSubmit = useCallback((): void => {
    clearTimers();
    setLogs([]);
    setRunning(true);

    let elapsed = 0;
    SIMULATED_PIPELINE.forEach((step: LogStep, index: number) => {
      elapsed += step.delayMs;
      const timer = setTimeout(() => {
        counter.current += 1;
        const entry: LogEntry = {
          id: counter.current,
          timestamp: formatTime(new Date()),
          level: step.level,
          message: step.message,
        };
        setLogs((prev: LogEntry[]) => [...prev, entry]);
        if (index === SIMULATED_PIPELINE.length - 1) {
          setRunning(false);
        }
      }, elapsed);
      timeouts.current.push(timer);
    });
  }, [clearTimers]);

  const handleReset = useCallback((): void => {
    clearTimers();
    setProfile("");
    setLogs([]);
    setRunning(false);
  }, [clearTimers]);

  return (
    <div className="min-h-screen bg-[#09090b] text-zinc-100">
      <TelemetryHeader />
      <main className="mx-auto grid max-w-7xl grid-cols-1 gap-6 px-6 py-6 lg:grid-cols-2">
        <IngressPanel
          value={profile}
          disabled={running}
          onChange={setProfile}
          onSubmit={handleSubmit}
          onReset={handleReset}
        />
        <LogTerminal logs={logs} running={running} />
      </main>
    </div>
  );
}