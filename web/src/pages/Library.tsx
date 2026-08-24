import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type Project } from "../api";

export default function Library() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [farm, setFarm] = useState<string[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api
      .projects()
      .then((data) => setProjects(data.projects))
      .catch((e) => setErr(String(e.message || e)));
    api
      .cartridges()
      .then((data) => setFarm(data.cartridges.filter((c) => c.ready).map((c) => c.id)))
      .catch(() => undefined);
  }, []);

  return (
    <div>
      <h1>Specialists</h1>
      <p className="lead">
        Each card is its own fine-tune, not a system prompt. Add a niche, distill, train, then ask the farm by
        name: <span className="mono">model: gopher-kungfu</span>
      </p>
      {err && <p className="err">{err}</p>}
      {!projects.length && !err && (
        <p className="muted">
          Empty farm. <Link to="/new">+ SLM</Link> to start a specialist.
        </p>
      )}
      <div className="grid">
        {projects.map((p) => (
          <Link key={p.slug} to={`/p/${p.slug}/topics`} className="card">
            <div className="row">
              <span className={`badge ${p.status}`}>{p.status}</span>
              {(p.jobs || []).filter((j) => j.status === "running" || j.status === "queued").map((j) => (
                <span key={j.id} className="badge running">{j.kind}</span>
              ))}
              {farm.includes(p.slug) && <span className="badge exported">on farm</span>}
            </div>
            <h3>{p.name}</h3>
            <div className="mono">{p.slug}</div>
            <div className="meta">
              {p.topics.map((t) => t.label).join(" · ") || "No topics yet"}
              <br />
              {p.base_model} · {p.distill.train_count} train examples
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
