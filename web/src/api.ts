export type TopicRef = { id: string; label: string; custom?: boolean };

export type Job = {
  id: string;
  kind: string;
  status: string;
  progress: number;
  error?: string | null;
  log?: string[];
  project_slug?: string;
  created_at?: string;
  updated_at?: string;
};

export type Project = {
  slug: string;
  name: string;
  base_model: "qwen3-1.7b" | "ministral-3b";
  teacher_preset: string;
  teacher_model: string;
  teacher_base_url: string;
  topics: TopicRef[];
  status: string;
  error?: string | null;
  has_api_key: boolean;
  curriculum_count: number;
  planned_examples: number;
  batch_available?: boolean;
  distill: { examples_per_topic: number; train_count: number; eval_count: number; use_batch?: boolean };
  train: {
    lora_r: number;
    lora_alpha: number;
    epochs: number;
    seq_len: number;
    batch_size: number;
    grad_accum: number;
    learning_rate: number;
  };
  cartridge_path?: string | null;
  jobs?: Job[];
};

export type CurriculumItem = {
  id: string;
  topic: string;
  subtopic: string;
  skill: string;
  difficulty: string;
  notes: string;
};

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      detail = await res.text();
    }
    throw new Error(detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

export type RuntimeInfo = {
  torch?: string | null;
  cuda_built?: string | null;
  cuda_available: boolean;
  device_name?: string | null;
  host_gpu?: string | null;
  hint?: string | null;
};

export const api = {
  catalog: () => req<{ categories: { id: string; label: string; topics: TopicRef[] }[] }>("/api/catalog"),
  runtime: () => req<RuntimeInfo>("/api/runtime"),
  teachers: () => req<{ presets: { id: string; label: string; base_url: string; model: string; notes: string }[] }>("/api/teachers"),
  baseModels: () => req<{ models: { id: string; label: string; vram_hint: string; default?: boolean }[] }>("/api/base-models"),
  cartridges: () => req<{ cartridges: { id: string; name: string; description: string; ready: boolean }[] }>("/api/cartridges"),
  projects: () => req<{ projects: Project[] }>("/api/projects"),
  project: (slug: string) => req<Project>(`/api/projects/${slug}`),
  create: (body: Record<string, unknown>) => req<Project>("/api/projects", { method: "POST", body: JSON.stringify(body) }),
  patch: (slug: string, body: Record<string, unknown>) =>
    req<Project>(`/api/projects/${slug}`, { method: "PATCH", body: JSON.stringify(body) }),
  curriculum: (slug: string) => req<{ items: CurriculumItem[] }>(`/api/projects/${slug}/curriculum`),
  saveCurriculum: (slug: string, items: CurriculumItem[]) =>
    req<{ items: CurriculumItem[] }>(`/api/projects/${slug}/curriculum`, {
      method: "PUT",
      body: JSON.stringify({ items }),
    }),
  generateCurriculum: (slug: string) => req<Job>(`/api/projects/${slug}/curriculum/generate`, { method: "POST" }),
  distill: (slug: string) => req<Job>(`/api/projects/${slug}/distill`, { method: "POST" }),
  train: (slug: string) => req<Job>(`/api/projects/${slug}/train`, { method: "POST" }),
  exportCartridge: (slug: string) => req<Job>(`/api/projects/${slug}/export`, { method: "POST" }),
  projectJobs: (slug: string) => req<{ jobs: Job[] }>(`/api/projects/${slug}/jobs`),
  job: (jobId: string) => req<Job>(`/api/jobs/${jobId}`),
  cancelJob: (jobId: string) => req<Job>(`/api/jobs/${jobId}/cancel`, { method: "POST" }),
};

export function watchJob(jobId: string, onEvent: (ev: { type: string; message?: string; progress?: number; status?: string; error?: string }) => void): () => void {
  const src = new EventSource(`/api/jobs/${jobId}/events`);
  src.onmessage = (event) => {
    const data = JSON.parse(event.data);
    onEvent(data);
    if (data.type === "done" || data.type === "error" || data.type === "cancelled") src.close();
  };
  return () => src.close();
}
