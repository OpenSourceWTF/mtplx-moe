import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Copy, Trash } from "lucide-react";
import { api } from "../lib/api";
import type { MutableSettings } from "../lib/types";
import { Card } from "./Card";
import { useDashboardStore } from "../state/store";

const MUTABLE_SETTINGS_KEYS: (keyof MutableSettings)[] = [
  "depth",
  "temperature",
  "top_p",
  "top_k",
  "presence_penalty",
  "max_response_tokens",
  "stream_interval",
  "enable_thinking",
  "reasoning_parser",
];

const DEBOUNCE_MS = 250;

export function ControlsSidebar() {
  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12 lg:col-span-7">
        <DefaultsCard />
      </div>
      <div className="col-span-12 lg:col-span-5">
        <RestartRequiredCard />
        <div className="mt-4">
          <AdminActions />
        </div>
      </div>
    </div>
  );
}

function DefaultsCard() {
  const settings = useDashboardStore((s) => s.settings);
  const [draft, setDraft] = useState<MutableSettings | null>(settings);
  const queryClient = useQueryClient();
  const mutation = useMutation({
    mutationFn: (payload: Partial<MutableSettings>) => api.postSettings(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["snapshot"] });
    },
  });
  const [lastApplied, setLastApplied] = useState<Partial<MutableSettings> | null>(null);

  useEffect(() => {
    if (settings) setDraft((prev) => prev ?? settings);
  }, [settings]);

  useEffect(() => {
    if (!draft || !settings) return;
    const diff: Partial<MutableSettings> = {};
    // Diff ONLY the mutable keys. `draft` is seeded from the full settings
    // GET payload at runtime, so Object.keys(draft) also yields the
    // informational keys (object-valued ones differ by reference after every
    // snapshot refresh) and the echo-back used to trip the server's
    // all-or-nothing unknown_settings 400 — the 2026-07-02
    // presence-penalty-persistence bug.
    MUTABLE_SETTINGS_KEYS.forEach((key) => {
      if (draft[key] !== settings[key]) {
        (diff as Record<string, unknown>)[key] = draft[key];
      }
    });
    if (Object.keys(diff).length === 0) return;
    const handle = window.setTimeout(() => {
      mutation.mutate(diff, {
        onSuccess: (res) => setLastApplied(res.applied),
      });
    }, DEBOUNCE_MS);
    return () => window.clearTimeout(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft]);

  if (!draft) {
    return (
      <Card title="Defaults" subtitle="loading server settings...">
        <div className="text-sm text-[var(--text-muted)]">
          Settings will appear once the dashboard receives its first snapshot.
        </div>
      </Card>
    );
  }

  return (
    <Card
      title="Defaults"
      subtitle="server-side defaults applied to every chat completion"
      action={
        lastApplied ? (
          <span className="text-xs text-[var(--text-muted)]">
            applied · {Object.keys(lastApplied).join(", ")}
          </span>
        ) : undefined
      }
    >
      <div className="space-y-4">
        <NumberField
          label="depth"
          value={draft.depth}
          min={0}
          max={5}
          onChange={(v) => setDraft({ ...draft, depth: v })}
        />
        <NumberField
          label="temperature"
          value={draft.temperature}
          min={0}
          max={2}
          step={0.05}
          onChange={(v) => setDraft({ ...draft, temperature: v })}
        />
        <NumberField
          label="top_p"
          value={draft.top_p}
          min={0}
          max={1}
          step={0.01}
          onChange={(v) => setDraft({ ...draft, top_p: v })}
        />
        <NumberField
          label="top_k"
          value={draft.top_k}
          min={0}
          max={2000}
          step={1}
          onChange={(v) => setDraft({ ...draft, top_k: v })}
        />
        <NumberField
          label="presence_penalty"
          value={draft.presence_penalty ?? 0}
          min={0}
          max={2}
          step={0.05}
          onChange={(v) => setDraft({ ...draft, presence_penalty: v })}
          description="0 is exact (best for coding); 0.5-1.5 discourages repetition."
        />
        <NumberField
          label="stream_interval"
          value={draft.stream_interval}
          min={1}
          max={32}
          step={1}
          onChange={(v) => setDraft({ ...draft, stream_interval: v })}
        />
        <ToggleField
          label="enable_thinking"
          value={draft.enable_thinking}
          onChange={(v) => setDraft({ ...draft, enable_thinking: v })}
          description="When on, requests default to including reasoning content."
        />
        <SelectField
          label="reasoning_parser"
          value={draft.reasoning_parser}
          options={["qwen3", "poolside_v1", "none"]}
          onChange={(v) => setDraft({ ...draft, reasoning_parser: v })}
        />
        {mutation.isError ? (
          <div className="text-xs text-[var(--accent-hot)]">
            {String((mutation.error as Error).message)}
          </div>
        ) : null}
      </div>
    </Card>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  description,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (next: number) => void;
  description?: string;
}) {
  return (
    <label className="block">
      <div className="flex items-baseline justify-between text-xs text-[var(--text-muted)]">
        <span>{label}</span>
        <span className="tabular-nums text-[var(--text-primary)]">{Number(value).toFixed(step < 1 ? 2 : 0)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full mt-1 accent-[var(--accent)]"
      />
      {description ? (
        <div className="mt-0.5 text-xs text-[var(--text-muted)]">{description}</div>
      ) : null}
    </label>
  );
}

function ToggleField({
  label,
  value,
  onChange,
  description,
}: {
  label: string;
  value: boolean;
  onChange: (next: boolean) => void;
  description?: string;
}) {
  return (
    <label className="flex items-start justify-between gap-3">
      <div>
        <div className="text-sm text-[var(--text-primary)]">{label}</div>
        {description ? (
          <div className="text-xs text-[var(--text-muted)]">{description}</div>
        ) : null}
      </div>
      <button
        type="button"
        onClick={() => onChange(!value)}
        className={`h-5 w-9 rounded-full transition-colors relative shrink-0 ${
          value ? "bg-[var(--accent)]" : "bg-[var(--border-soft)]"
        }`}
        aria-pressed={value}
      >
        <span
          className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-transform ${
            value ? "translate-x-4" : "translate-x-0.5"
          }`}
        />
      </button>
    </label>
  );
}

function SelectField({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (next: string) => void;
}) {
  return (
    <label className="block">
      <div className="text-xs text-[var(--text-muted)]">{label}</div>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1 w-full bg-[var(--bg-elevated)] border border-[var(--border-soft)] rounded px-2 py-1.5 text-sm text-[var(--text-primary)]"
      >
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </label>
  );
}

function RestartRequiredCard() {
  const modelId = useDashboardStore((s) => s.modelId);
  const profileName = useDashboardStore((s) => s.profileName);
  const restartCommand = `mtplx serve --model ${modelId ?? "<model>"} --profile ${profileName ?? "<profile>"} --port 8000`;
  const copy = () => {
    if (typeof navigator !== "undefined") {
      navigator.clipboard?.writeText(restartCommand);
    }
  };
  return (
    <Card
      title="Restart required"
      subtitle="profile · model · MTP · host · port can only change at startup"
    >
      <p className="text-xs text-[var(--text-muted)] mb-3">
        These settings live on <code>state.args</code> but require a model
        reload to take effect. The dashboard refuses to mutate them through
        the live settings endpoint. Copy the CLI command instead.
      </p>
      <div className="rounded border border-[var(--border-soft)] bg-[var(--bg-elevated)] px-3 py-2 font-mono text-xs text-[var(--text-primary)] overflow-x-auto">
        {restartCommand}
      </div>
      <button
        type="button"
        onClick={copy}
        className="mt-3 inline-flex items-center gap-1.5 text-xs text-[var(--accent-cool)] hover:text-[var(--accent)]"
      >
        <Copy className="size-3.5" />
        copy restart command
      </button>
    </Card>
  );
}

function AdminActions() {
  const queryClient = useQueryClient();
  const [confirming, setConfirming] = useState(false);
  const clearAll = useMutation({
    mutationFn: () => api.postClearCache(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
      setConfirming(false);
    },
  });
  return (
    <Card title="Admin actions" subtitle="bank-wide controls">
      <button
        type="button"
        onClick={() => setConfirming(true)}
        className="inline-flex items-center gap-2 text-sm text-[var(--accent-hot)] hover:bg-[var(--accent-hot)]/10 rounded px-3 py-2 transition-colors"
      >
        <Trash className="size-4" />
        Clear all SessionBank entries
      </button>
      {confirming ? (
        <div className="mt-3 p-3 rounded-md border border-[var(--accent-hot)]/40 bg-[var(--accent-hot)]/5 text-sm text-[var(--text-primary)]">
          <p>
            Evict every cached prefix? Future requests will pay full prefill
            until the cache refills.
          </p>
          <div className="mt-3 flex gap-2">
            <button
              type="button"
              onClick={() => clearAll.mutate()}
              disabled={clearAll.isPending}
              className="text-xs px-3 py-1 rounded bg-[var(--accent-hot)] text-white disabled:opacity-50"
            >
              {clearAll.isPending ? "Clearing..." : "Yes, clear cache"}
            </button>
            <button
              type="button"
              onClick={() => setConfirming(false)}
              className="text-xs px-3 py-1 rounded border border-[var(--border-soft)] text-[var(--text-muted)]"
            >
              Cancel
            </button>
          </div>
        </div>
      ) : null}
    </Card>
  );
}
