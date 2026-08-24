import { useEffect, useState } from "react";
import { api } from "../api";

export type SetupValues = {
  name: string;
  slug: string;
  base_model: string;
  teacher_preset: string;
  teacher_model: string;
  teacher_base_url: string;
  api_key: string;
};

type Props = {
  values: SetupValues;
  onChange: (patch: Partial<SetupValues>) => void;
  slugLocked?: boolean;
  hasApiKey?: boolean;
  fillDefaults?: boolean;
};

export default function SetupForm({ values, onChange, slugLocked, hasApiKey, fillDefaults }: Props) {
  const [presets, setPresets] = useState<{ id: string; label: string; base_url: string; model: string; notes: string }[]>([]);
  const [bases, setBases] = useState<{ id: string; label: string; vram_hint: string }[]>([]);

  useEffect(() => {
    let cancelled = false;
    api.teachers().then((d) => {
      if (cancelled) return;
      setPresets(d.presets);
      if (!fillDefaults) return;
      const first = d.presets.find((p) => p.id === values.teacher_preset) || d.presets[0];
      if (first) {
        onChange({
          teacher_preset: first.id,
          teacher_model: first.model,
          teacher_base_url: first.base_url,
        });
      }
    });
    api.baseModels().then((d) => {
      if (!cancelled) setBases(d.models);
    });
    return () => {
      cancelled = true;
    };
    // Load catalogs once. Create-form defaults are filled from the first response.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function applyPreset(id: string) {
    const found = presets.find((p) => p.id === id);
    onChange({
      teacher_preset: id,
      ...(found ? { teacher_model: found.model, teacher_base_url: found.base_url } : {}),
    });
  }

  const notes = presets.find((p) => p.id === values.teacher_preset)?.notes;

  return (
    <>
      <label>
        Display name
        <input
          value={values.name}
          onChange={(e) => onChange({ name: e.target.value })}
          placeholder="Gopher Kungfu"
          required
        />
      </label>
      <label>
        Slug (OpenAI model id)
        <input
          value={values.slug}
          onChange={(e) => onChange({ slug: e.target.value })}
          placeholder="gopher-kungfu"
          disabled={slugLocked}
          title={slugLocked ? "Slug is the farm model id and cannot change after create." : undefined}
        />
      </label>
      <label>
        Base model
        <select value={values.base_model} onChange={(e) => onChange({ base_model: e.target.value })}>
          {bases.map((b) => (
            <option key={b.id} value={b.id}>
              {b.label} — {b.vram_hint}
            </option>
          ))}
        </select>
      </label>
      <label>
        Teacher
        <select value={values.teacher_preset} onChange={(e) => applyPreset(e.target.value)}>
          {presets.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
        </select>
      </label>
      <label>
        Teacher model
        <input value={values.teacher_model} onChange={(e) => onChange({ teacher_model: e.target.value })} />
      </label>
      <label>
        Inference base URL
        <input value={values.teacher_base_url} onChange={(e) => onChange({ teacher_base_url: e.target.value })} />
      </label>
      <label>
        API key (stored locally, never in project.json)
        <input
          type="password"
          value={values.api_key}
          onChange={(e) => onChange({ api_key: e.target.value })}
          autoComplete="off"
          placeholder={hasApiKey ? "Saved — leave blank to keep" : ""}
        />
      </label>
      {notes && <div className="note">{notes} Some teacher APIs restrict training on outputs — that is your responsibility.</div>}
    </>
  );
}
