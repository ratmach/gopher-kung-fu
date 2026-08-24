import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import SetupForm, { type SetupValues } from "../components/SetupForm";
import { api } from "../api";

const empty: SetupValues = {
  name: "",
  slug: "",
  base_model: "qwen3-1.7b",
  teacher_preset: "deepseek",
  teacher_model: "",
  teacher_base_url: "",
  api_key: "",
};

export default function NewSlm() {
  const nav = useNavigate();
  const [values, setValues] = useState<SetupValues>(empty);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      const created = await api.create({
        name: values.name,
        slug: values.slug || undefined,
        base_model: values.base_model,
        teacher_preset: values.teacher_preset,
        teacher_model: values.teacher_model,
        teacher_base_url: values.teacher_base_url,
        api_key: values.api_key || undefined,
      });
      nav(`/p/${created.slug}/topics`);
    } catch (ex) {
      setErr(String((ex as Error).message));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <h1>+ SLM</h1>
      <p className="lead">Name a specialist and point at a teacher that will write its curriculum and synthetic coding data.</p>
      <form className="form" onSubmit={submit}>
        <SetupForm
          values={values}
          onChange={(patch) => setValues((cur) => ({ ...cur, ...patch }))}
          fillDefaults
        />
        {err && <p className="err">{err}</p>}
        <div className="row">
          <button className="btn" disabled={busy || !values.name}>
            Create specialist
          </button>
        </div>
      </form>
    </div>
  );
}
