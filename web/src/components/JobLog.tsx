import { useEffect, useRef, useState } from "react";
import { api, watchJob } from "../api";

type Props = {
  jobId: string | null;
  running?: boolean;
  onDone?: (ok: boolean) => void;
  onCancel?: () => void;
};

export default function JobLog({ jobId, running, onDone, onCancel }: Props) {
  const [lines, setLines] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<string>(running ? "running" : "");
  const [busy, setBusy] = useState(false);
  const box = useRef<HTMLPreElement>(null);
  const doneRef = useRef(onDone);
  doneRef.current = onDone;

  useEffect(() => {
    if (!jobId) return;
    setLines([]);
    setProgress(0);
    const stop = watchJob(jobId, (ev) => {
      const msg = ev.message;
      if (msg) setLines((cur) => [...cur, msg]);
      if (typeof ev.progress === "number") setProgress(ev.progress);
      if (ev.status) setStatus(ev.status);
      if (ev.type === "done" || ev.type === "error" || ev.type === "cancelled") {
        setStatus(ev.type === "done" ? "done" : ev.type);
        doneRef.current?.(ev.type === "done");
      }
    });
    return stop;
  }, [jobId]);

  useEffect(() => {
    if (box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [lines]);

  async function cancel() {
    if (!jobId) return;
    setBusy(true);
    try {
      if (onCancel) await onCancel();
      else await api.cancelJob(jobId);
    } finally {
      setBusy(false);
    }
  }

  if (!jobId) return null;
  const active = status === "running" || status === "queued" || running;
  return (
    <div>
      <div className="row" style={{ marginTop: 12 }}>
        {status && <span className={`badge ${status === "error" || status === "cancelled" ? "error" : status}`}>{status}</span>}
        <span className="muted">Job {jobId} runs on the server. You can leave this page.</span>
        {active && (
          <button type="button" className="btn danger" disabled={busy} onClick={cancel}>
            Cancel
          </button>
        )}
      </div>
      <div className="bar">
        <i style={{ width: `${Math.round(progress * 100)}%` }} />
      </div>
      <pre className="log" ref={box}>
        {lines.join("\n") || "Waiting for worker…"}
      </pre>
    </div>
  );
}
