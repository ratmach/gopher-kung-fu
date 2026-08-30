import { useCallback, useEffect, useState, type FormEvent } from "react";
import { NavLink, Route, Routes, useParams } from "react-router-dom";
import Hint from "../components/Hint";
import JobLog from "../components/JobLog";
import SetupForm, { type SetupValues } from "../components/SetupForm";
import { api, type CurriculumItem, type Job, type Project, type RuntimeInfo, type TopicRef } from "../api";

function latestJob(project: Project, kind: string): Job | undefined {
  return (project.jobs || []).find((job) => job.kind === kind);
}

function isActive(job?: Job) {
  return job?.status === "running" || job?.status === "queued";
}

function useStepJob(project: Project, kind: string, onChange: () => void) {
  const latest = latestJob(project, kind);
  const [jobId, setJobId] = useState<string | null>(latest?.id ?? null);
  const [running, setRunning] = useState(isActive(latest));

  useEffect(() => {
    const next = latestJob(project, kind);
    if (next) {
      setJobId(next.id);
      setRunning(isActive(next));
    }
  }, [project.jobs, kind]);

  async function start(factory: () => Promise<Job>) {
    const job = await factory();
    setJobId(job.id);
    setRunning(true);
  }

  async function cancel() {
    if (!jobId) return;
    await api.cancelJob(jobId);
    setRunning(false);
    onChange();
  }

  function handleDone() {
    setRunning(false);
    onChange();
  }

  return { jobId, running, start, cancel, handleDone };
}

function stepClass({ isActive }: { isActive: boolean }) {
  return isActive ? "on" : "";
}

export default function Wizard() {
  const { slug = "" } = useParams();
  const [project, setProject] = useState<Project | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const reload = useCallback(() => {
    api.project(slug).then(setProject).catch((e) => setErr(String(e.message || e)));
  }, [slug]);

  useEffect(() => {
    reload();
  }, [reload]);

  if (err) return <p className="err">{err}</p>;
  if (!project) return <p className="muted">Loading…</p>;

  return (
    <div>
      <h1>{project.name}</h1>
      <p className="mono muted">{project.slug} · {project.base_model} · {project.status}</p>
      {(project.jobs || []).some((j) => isActive(j)) && (
        <p className="note">
          {(project.jobs || [])
            .filter(isActive)
            .map((j) => `${j.kind} ${j.id}`)
            .join(" · ")}{" "}
          is running in the background.
        </p>
      )}
      {project.error && <p className="err">{project.error}</p>}
      <nav className="steps">
        <NavLink to="setup" className={stepClass}>Setup</NavLink>
        <NavLink to="topics" className={stepClass}>Topics</NavLink>
        <NavLink to="curriculum" className={stepClass}>Curriculum</NavLink>
        <NavLink to="distill" className={stepClass}>Distill</NavLink>
        <NavLink to="train" className={stepClass}>Train</NavLink>
        <NavLink to="export" className={stepClass}>Export</NavLink>
      </nav>
      <Routes>
        <Route path="setup" element={<SetupStep project={project} onSaved={setProject} />} />
        <Route path="topics" element={<TopicsStep project={project} onSaved={setProject} />} />
        <Route path="curriculum" element={<CurriculumStep project={project} onChange={reload} />} />
        <Route path="distill" element={<DistillStep project={project} onChange={reload} />} />
        <Route path="train" element={<TrainStep project={project} onChange={reload} />} />
        <Route path="export" element={<ExportStep project={project} onChange={reload} />} />
        <Route path="*" element={<TopicsStep project={project} onSaved={setProject} />} />
      </Routes>
    </div>
  );
}

function fromProject(project: Project): SetupValues {
  return {
    name: project.name,
    slug: project.slug,
    base_model: project.base_model,
    teacher_preset: project.teacher_preset,
    teacher_model: project.teacher_model,
    teacher_base_url: project.teacher_base_url,
    api_key: "",
  };
}

function SetupStep({ project, onSaved }: { project: Project; onSaved: (p: Project) => void }) {
  const [values, setValues] = useState<SetupValues>(() => fromProject(project));
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const jobsRunning = (project.jobs || []).some(isActive);

  useEffect(() => {
    setValues(fromProject(project));
  }, [
    project.slug,
    project.name,
    project.base_model,
    project.teacher_preset,
    project.teacher_model,
    project.teacher_base_url,
  ]);

  async function save(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    setMsg(null);
    try {
      const next = await api.patch(project.slug, {
        name: values.name,
        base_model: values.base_model,
        teacher_preset: values.teacher_preset,
        teacher_model: values.teacher_model,
        teacher_base_url: values.teacher_base_url,
        ...(values.api_key.trim() ? { api_key: values.api_key.trim() } : {}),
      });
      onSaved(next);
      setValues({ ...fromProject(next), api_key: "" });
      setMsg("Setup saved.");
    } catch (ex) {
      setErr(String((ex as Error).message));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <p className="lead">
        Change the teacher, model, or API key anytime. The slug is the farm model id and stays fixed.
      </p>
      <form className="form" onSubmit={save}>
        <SetupForm
          values={values}
          onChange={(patch) => setValues((cur) => ({ ...cur, ...patch }))}
          slugLocked
          hasApiKey={project.has_api_key}
        />
        {jobsRunning && <p className="note">A job is running. Saved changes apply to the next curriculum or distill run.</p>}
        {err && <p className="err">{err}</p>}
        <div className="row">
          <button className="btn" disabled={busy || !values.name}>
            Save setup
          </button>
          {msg && <span className="muted">{msg}</span>}
        </div>
      </form>
    </div>
  );
}

function TopicsStep({ project, onSaved }: { project: Project; onSaved: (p: Project) => void }) {
  const [catalog, setCatalog] = useState<{ id: string; label: string; topics: TopicRef[] }[]>([]);
  const [selected, setSelected] = useState<TopicRef[]>(project.topics);
  const [custom, setCustom] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  useEffect(() => {
    api.catalog().then((d) => setCatalog(d.categories));
  }, []);

  function toggle(topic: TopicRef) {
    setSelected((cur) =>
      cur.some((t) => t.id === topic.id) ? cur.filter((t) => t.id !== topic.id) : [...cur, { ...topic, custom: false }],
    );
  }

  function addCustom() {
    const label = custom.trim();
    if (!label) return;
    const id = "custom:" + label.toLowerCase().replace(/[^a-z0-9]+/g, "-");
    if (!selected.some((t) => t.id === id)) setSelected([...selected, { id, label, custom: true }]);
    setCustom("");
  }

  async function save() {
    setBusy(true);
    try {
      const next = await api.patch(project.slug, { topics: selected });
      onSaved(next);
      setMsg("Topics saved. Keep the niche narrow.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <p className="lead">Pick a tight niche. Wide “knows everything” specialists are out of scope.</p>
      {catalog.map((cat) => (
        <div key={cat.id} style={{ marginBottom: 16 }}>
          <h3>{cat.label}</h3>
          <div className="chips">
            {cat.topics.map((t) => (
              <button
                key={t.id}
                type="button"
                className={`chip ${selected.some((s) => s.id === t.id) ? "on" : ""}`}
                onClick={() => toggle(t)}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>
      ))}
      <div className="row">
        <input value={custom} onChange={(e) => setCustom(e.target.value)} placeholder="Custom topic, e.g. sqlc" />
        <button type="button" className="btn ghost" onClick={addCustom}>
          Add custom
        </button>
      </div>
      {selected.filter((t) => t.custom).length > 0 && (
        <p className="muted">Custom: {selected.filter((t) => t.custom).map((t) => t.label).join(", ")}</p>
      )}
      <div className="row" style={{ marginTop: 16 }}>
        <button className="btn" disabled={busy} onClick={save}>
          Save topics
        </button>
        {msg && <span className="muted">{msg}</span>}
      </div>
    </div>
  );
}

function CurriculumStep({ project, onChange }: { project: Project; onChange: () => void }) {
  const [items, setItems] = useState<CurriculumItem[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [itemsPerTopic, setItemsPerTopic] = useState(project.items_per_topic ?? 12);
  const job = useStepJob(project, "curriculum", onChange);
  const topicCount = project.topics.length;
  const plannedItems = itemsPerTopic * topicCount;

  const load = useCallback(() => {
    api.curriculum(project.slug).then((d) => setItems(d.items));
  }, [project.slug]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    setItemsPerTopic(project.items_per_topic ?? 12);
  }, [project.items_per_topic]);

  async function generate() {
    setErr(null);
    try {
      await job.start(() => api.generateCurriculum(project.slug, itemsPerTopic));
    } catch (e) {
      setErr(String((e as Error).message));
    }
  }

  const [saved, setSaved] = useState<string | null>(null);

  async function save() {
    setErr(null);
    await api.saveCurriculum(project.slug, items);
    setSaved(`Saved ${items.length} syllabus item${items.length === 1 ? "" : "s"}.`);
    onChange();
  }

  function update(index: number, patch: Partial<CurriculumItem>) {
    setItems((cur) => cur.map((item, i) => (i === index ? { ...item, ...patch } : item)));
    setSaved(null);
  }

  function addItem() {
    const topic = project.topics[0]?.label || "";
    setItems((cur) => [
      ...cur,
      {
        id: `custom-${Date.now().toString(36)}-${cur.length + 1}`,
        topic,
        subtopic: "",
        skill: "write",
        difficulty: "medium",
        notes: "",
      },
    ]);
    setSaved(null);
  }

  function removeItem(index: number) {
    setItems((cur) => cur.filter((_, i) => i !== index));
    setSaved(null);
  }

  return (
    <div>
      <p className="lead">
        Set how many syllabus items the teacher should write for each selected topic, then generate. More items means denser coverage of the same niche.
      </p>
      <label>
        <span className="field-label">
          Items per topic
          <Hint text="How many syllabus rows to fetch for each topic you saved on the Topics step. 12 is a typical specialist; 30–80 is a deeper niche before distill." />
        </span>
        <input
          type="number"
          min={4}
          max={80}
          value={itemsPerTopic}
          disabled={job.running}
          onChange={(e) => setItemsPerTopic(Number(e.target.value))}
        />
      </label>
      <p className="muted">
        {topicCount
          ? `${itemsPerTopic} × ${topicCount} topic${topicCount === 1 ? "" : "s"} → ${plannedItems} syllabus items`
          : "Save at least one topic first."}
      </p>
      <div className="row">
        <button className="btn" disabled={job.running || !topicCount} onClick={generate}>
          Generate syllabus
        </button>
        <button className="btn ghost" type="button" onClick={addItem} disabled={job.running}>
          Add item
        </button>
        <button className="btn ghost" onClick={save} disabled={job.running}>
          Save edits
        </button>
        {saved && <span className="muted">{saved}</span>}
      </div>
      {err && <p className="err">{err}</p>}
      <JobLog jobId={job.jobId} running={job.running} onCancel={job.cancel} onDone={() => { load(); job.handleDone(); }} />
      <table>
        <thead>
          <tr>
            <th>Topic</th>
            <th>Subtopic</th>
            <th>Skill</th>
            <th>Difficulty</th>
            <th>Notes</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {items.map((item, i) => (
            <tr key={item.id}>
              <td>
                <input value={item.topic} onChange={(e) => update(i, { topic: e.target.value })} placeholder="Go" />
              </td>
              <td>
                <input value={item.subtopic} onChange={(e) => update(i, { subtopic: e.target.value })} placeholder="interfaces" />
              </td>
              <td>
                <select value={item.skill} onChange={(e) => update(i, { skill: e.target.value })}>
                  {["write", "review", "debug", "refactor", "idiom"].map((s) => (
                    <option key={s}>{s}</option>
                  ))}
                </select>
              </td>
              <td>
                <select value={item.difficulty} onChange={(e) => update(i, { difficulty: e.target.value })}>
                  {["easy", "medium", "hard"].map((s) => (
                    <option key={s}>{s}</option>
                  ))}
                </select>
              </td>
              <td>
                <input value={item.notes} onChange={(e) => update(i, { notes: e.target.value })} placeholder="What this item should teach" />
              </td>
              <td>
                <button type="button" className="btn ghost btn-icon" onClick={() => removeItem(i)} aria-label="Remove item">
                  ×
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {!items.length && <p className="muted">Empty syllabus. Add an item or generate one.</p>}
    </div>
  );
}

function DistillStep({ project, onChange }: { project: Project; onChange: () => void }) {
  const [perTopic, setPerTopic] = useState(project.distill.examples_per_topic);
  const [useBatch, setUseBatch] = useState(Boolean(project.distill.use_batch));
  const [err, setErr] = useState<string | null>(null);
  const job = useStepJob(project, "distill", onChange);
  const batchOk = Boolean(project.batch_available);
  const canToggleBatch = batchOk || useBatch;

  async function run() {
    setErr(null);
    try {
      await api.patch(project.slug, {
        distill: { ...project.distill, examples_per_topic: perTopic, use_batch: useBatch },
      });
      await job.start(() => api.distill(project.slug));
    } catch (e) {
      setErr(String((e as Error).message));
    }
  }

  return (
    <div>
      <p className="lead">
        Synthetic coding data is stored as ShareGPT JSONL, split by topic for reuse
        (<span className="mono">data/library/&lt;topic&gt;</span> plus this project's{" "}
        <span className="mono">synthetic/topics</span>). Assistant replies are filtered to be
        code-first (spec in, full file out). Default is {project.planned_examples || perTopic} planned
        examples. Hold-out eval is 10%. A re-run fills syllabus items that are still short (a new topic
        or new rows) and only clears the inbox when every current item is already at quota.
      </p>
      <label>
        Examples per topic
        <input type="number" min={8} max={400} value={perTopic} onChange={(e) => setPerTopic(Number(e.target.value))} />
      </label>
      <label className="check">
        <input
          type="checkbox"
          checked={useBatch}
          disabled={!canToggleBatch}
          onChange={(e) => setUseBatch(e.target.checked)}
        />
        <span>
          <span className="field-label">
            Use OpenRouter batch
            <Hint text="Submits distillation as one cheaper async batch (~50% off live rates). Results can take minutes to hours. Leave this app running until the job finishes. OpenRouter only." />
          </span>
          <span className="muted">
            {batchOk
              ? "Cheaper, slower. Good when you do not need the dataset immediately."
              : "Available when the teacher is OpenRouter."}
          </span>
        </span>
      </label>
      {useBatch && !batchOk && (
        <p className="err">Batch is on, but this teacher is not OpenRouter. Uncheck it or switch the teacher.</p>
      )}
      <p className="muted">
        Current store: {project.distill.train_count} train / {project.distill.eval_count} eval
      </p>
      <button className="btn" disabled={job.running} onClick={run}>
        Distill
      </button>
      {err && <p className="err">{err}</p>}
      <JobLog jobId={job.jobId} running={job.running} onCancel={job.cancel} onDone={job.handleDone} />
    </div>
  );
}

function TrainStep({ project, onChange }: { project: Project; onChange: () => void }) {
  const [train, setTrain] = useState(project.train);
  const [err, setErr] = useState<string | null>(null);
  const [runtime, setRuntime] = useState<RuntimeInfo | null>(null);
  const job = useStepJob(project, "train", onChange);

  useEffect(() => {
    setTrain(project.train);
  }, [project.train]);

  useEffect(() => {
    api.runtime().then(setRuntime).catch(() => setRuntime(null));
  }, []);

  async function run() {
    setErr(null);
    try {
      await api.patch(project.slug, { train });
      await job.start(() => api.train(project.slug));
    } catch (e) {
      setErr(String((e as Error).message));
    }
  }

  return (
    <div>
      <p className="lead">
        Automated QLoRA via Unsloth. Needs an NVIDIA GPU and a CUDA PyTorch build in this venv.
        First start can sit idle for a minute while kernels compile — leave it running.
      </p>
      {runtime && (
        <p className={runtime.cuda_available ? "note" : "err"}>
          {runtime.cuda_available
            ? `GPU: ${runtime.device_name} · ${runtime.torch} · CUDA ${runtime.cuda_built}`
            : runtime.hint || "PyTorch cannot see a GPU."}
        </p>
      )}
      <div className="form">
        <label>
          <span className="field-label">
            LoRA rank
            <Hint text="How wide the adapter is. Higher learns more of the niche but uses more VRAM and can overfit. 8–16 is typical; 32 is the next step on 7B if 16 is shy." />
          </span>
          <input type="number" min={1} value={train.lora_r} onChange={(e) => setTrain({ ...train, lora_r: Number(e.target.value) })} />
        </label>
        <label>
          <span className="field-label">
            LoRA alpha
            <Hint text="How strongly the adapter is mixed back into the base model. Keep it equal to rank, or 2× rank if the fine-tune is too shy." />
          </span>
          <input type="number" min={1} value={train.lora_alpha} onChange={(e) => setTrain({ ...train, lora_alpha: Number(e.target.value) })} />
        </label>
        <label>
          <span className="field-label">
            Epochs
            <Hint text="How many times the trainer walks the distilled JSONL. 1–2 is enough for a few hundred examples. More than 3 usually overfits a small set." />
          </span>
          <input type="number" min={0.5} step="0.5" value={train.epochs} onChange={(e) => setTrain({ ...train, epochs: Number(e.target.value) })} />
        </label>
        <label>
          <span className="field-label">
            Sequence length
            <Hint text="Max tokens per example (prompt plus answer). Match implement_go: 4096 for 3B/4B/7B/Lite. 2048 truncates complete files. Longer needs more VRAM." />
          </span>
          <input type="number" min={256} step={256} value={train.seq_len} onChange={(e) => setTrain({ ...train, seq_len: Number(e.target.value) })} />
        </label>
        <label>
          <span className="field-label">
            Learning rate
            <Hint text="Unsloth QLoRA default is 2e-4 (1.7B). Use 1e-4 on 3B/4B/7B/Lite so the adapter does not overshoot." />
          </span>
          <input
            type="number"
            min={0}
            step="any"
            value={train.learning_rate}
            onChange={(e) => setTrain({ ...train, learning_rate: Number(e.target.value) })}
          />
        </label>
        <label>
          <span className="field-label">
            Batch size
            <Hint text="Per-GPU batch. Keep 1 on 12 GB with seq 4096. Only raise if VRAM still has headroom." />
          </span>
          <input type="number" min={1} value={train.batch_size} onChange={(e) => setTrain({ ...train, batch_size: Number(e.target.value) })} />
        </label>
        <label>
          <span className="field-label">
            Gradient accumulation
            <Hint text="Effective batch is batch × this. Default 8. Cut to 4 if sequence 4096 runs out of VRAM." />
          </span>
          <input type="number" min={1} value={train.grad_accum} onChange={(e) => setTrain({ ...train, grad_accum: Number(e.target.value) })} />
        </label>
        <label>
          <span className="field-label">
            Warmup ratio
            <Hint text="Fraction of steps spent ramping the learning rate. 0.05 is typical; 0.03–0.1 is the useful range." />
          </span>
          <input
            type="number"
            min={0}
            max={0.5}
            step="0.01"
            value={train.warmup_ratio}
            onChange={(e) => setTrain({ ...train, warmup_ratio: Number(e.target.value) })}
          />
        </label>
        <label>
          <span className="field-label">
            Eval every N steps
            <Hint text="0 = once per epoch when eval.jsonl exists. Greater than 0 evaluates every N optimizer steps. No eval file means this is ignored." />
          </span>
          <input type="number" min={0} value={train.eval_steps} onChange={(e) => setTrain({ ...train, eval_steps: Number(e.target.value) })} />
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={train.train_on_responses_only !== false}
            onChange={(e) => setTrain({ ...train, train_on_responses_only: e.target.checked })}
          />
          <span>
            <span className="field-label">
              Train on responses only
              <Hint text="Loss on the assistant Go, not the spec. Leave on. Falls back to full-sequence loss if this Unsloth build cannot mask prompts." />
            </span>
            <span className="muted">Masks the user turn so the adapter learns to write files, not copy the prompt.</span>
          </span>
        </label>
      </div>
      <div className="row" style={{ marginTop: 14 }}>
        <button className="btn" disabled={job.running || project.distill.train_count < 4} onClick={run}>
          Start QLoRA
        </button>
      </div>
      {err && <p className="err">{err}</p>}
      <JobLog jobId={job.jobId} running={job.running} onCancel={job.cancel} onDone={job.handleDone} />
    </div>
  );
}

function ExportStep({ project, onChange }: { project: Project; onChange: () => void }) {
  const [err, setErr] = useState<string | null>(null);
  const job = useStepJob(project, "export", onChange);

  async function run() {
    setErr(null);
    try {
      await job.start(() => api.exportCartridge(project.slug));
    } catch (e) {
      setErr(String((e as Error).message));
    }
  }

  return (
    <div>
      <p className="lead">Merge is already done after train. This step writes Q4_K_M (card default) plus Q5_K_M beside it into cartridges/.</p>
      {project.cartridge_path && <p className="mono">{project.cartridge_path}</p>}
      <div className="note">
        Serve the farm:{" "}
        <span className="mono">go run ./server/cmd/cartridge-server -cartridges ./cartridges</span>
        <br />
        Then ask <span className="mono">POST /v1/chat/completions</span> with{" "}
        <span className="mono">model: {project.slug}</span>
      </div>
      <button className="btn" disabled={job.running} onClick={run} style={{ marginTop: 12 }}>
        Export GGUF
      </button>
      {err && <p className="err">{err}</p>}
      <JobLog jobId={job.jobId} running={job.running} onCancel={job.cancel} onDone={job.handleDone} />
    </div>
  );
}
