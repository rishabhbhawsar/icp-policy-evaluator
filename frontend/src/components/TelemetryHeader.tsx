// src/components/TelemetryHeader.tsx
import type { FC } from "react";

export type AccentTone = "emerald" | "cyan" | "amber";

export interface MetricCardProps {
  label: string;
  value: string;
  caption: string;
  tone: AccentTone;
}

export interface CacheStatusBadgeProps {
  label: string;
  efficiency: string;
  active: boolean;
}

export interface TelemetryHeaderProps {
  precision?: string;
  recall?: string;
  cacheEfficiency?: string;
  cacheActive?: boolean;
}

const TONE_TEXT: Record<AccentTone, string> = {
  emerald: "text-emerald-400",
  cyan: "text-cyan-400",
  amber: "text-amber-400",
};

const TONE_DOT: Record<AccentTone, string> = {
  emerald: "bg-emerald-400",
  cyan: "bg-cyan-400",
  amber: "bg-amber-400",
};

const MetricCard: FC<MetricCardProps> = ({ label, value, caption, tone }) => (
  <div className="flex flex-col gap-2 rounded-md border border-[#27272a] bg-[#09090b] p-5">
    <div className="flex items-center gap-2">
      <span className={`h-1.5 w-1.5 rounded-full ${TONE_DOT[tone]}`} />
      <span className="font-mono text-[11px] uppercase tracking-widest text-zinc-500">
        {label}
      </span>
    </div>
    <span
      className={`font-mono text-3xl font-semibold tabular-nums ${TONE_TEXT[tone]}`}
    >
      {value}
    </span>
    <span className="text-xs text-zinc-500">{caption}</span>
  </div>
);

const CacheStatusBadge: FC<CacheStatusBadgeProps> = ({
  label,
  efficiency,
  active,
}) => (
  <div className="flex flex-col gap-2 rounded-md border border-[#27272a] bg-[#09090b] p-5">
    <div className="flex items-center gap-2">
      <span className="relative flex h-2 w-2">
        {active && (
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />
        )}
        <span
          className={`relative inline-flex h-2 w-2 rounded-full ${
            active ? "bg-emerald-400" : "bg-zinc-600"
          }`}
        />
      </span>
      <span className="font-mono text-[11px] uppercase tracking-widest text-zinc-500">
        Cache Layer
      </span>
    </div>
    <div className="flex items-center gap-3">
      <span
        className={`inline-flex items-center rounded border px-2.5 py-1 font-mono text-xs font-medium ${
          active
            ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
            : "border-[#27272a] bg-zinc-900 text-zinc-500"
        }`}
      >
        {active ? "ACTIVE" : "OFFLINE"}
      </span>
      <span className="font-mono text-sm text-zinc-200">{label}</span>
    </div>
    <span className="font-mono text-xs text-emerald-400">
      ({efficiency} Efficiency)
    </span>
  </div>
);

const TelemetryHeader: FC<TelemetryHeaderProps> = ({
  precision = "1.000",
  recall = "0.778",
  cacheEfficiency = "100%",
  cacheActive = true,
}) => {
  return (
    <header className="w-full border-b border-[#27272a] bg-[#09090b]">
      <div className="mx-auto flex max-w-7xl flex-col gap-6 px-6 py-6">
        <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="font-mono text-[11px] uppercase tracking-widest text-cyan-400">
              icp-policy-console
            </p>
            <h1 className="text-xl font-semibold tracking-tight text-zinc-100">
              Compliance Audit Engine
            </h1>
          </div>
          <p className="font-mono text-xs text-zinc-500">
            Async LLM-as-a-Judge &middot; Live Telemetry
          </p>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <MetricCard
            label="Precision"
            value={precision}
            caption="Validated against benchmark set"
            tone="emerald"
          />
          <MetricCard
            label="Recall"
            value={recall}
            caption="Validated against benchmark set"
            tone="cyan"
          />
          <CacheStatusBadge
            label="SQLite WAL Caching"
            efficiency={cacheEfficiency}
            active={cacheActive}
          />
        </div>
      </div>
    </header>
  );
};

export default TelemetryHeader;