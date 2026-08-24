import { Link, Navigate, Route, Routes } from "react-router-dom";
import Library from "./pages/Library";
import NewSlm from "./pages/NewSlm";
import Wizard from "./pages/Wizard";

export default function App() {
  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <Link to="/">Specialist farm</Link>
          <small>True niches. Named weights. One local server.</small>
        </div>
        <Link className="btn" to="/new">
          + SLM
        </Link>
      </header>
      <Routes>
        <Route path="/" element={<Library />} />
        <Route path="/new" element={<NewSlm />} />
        <Route path="/p/:slug/*" element={<Wizard />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}
