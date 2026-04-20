/* SynapseMem — minimal live graph viewer.
 *
 * Talks to the backend on the same origin. No build step: plain JS + cytoscape
 * loaded from a CDN. Good enough for v0.1 demos; a proper React/Vite frontend
 * with live streaming comes in a later milestone.
 */

const STYLE = [
  {
    selector: "node",
    style: {
      label: "data(label)",
      "font-size": 10,
      "text-valign": "center",
      "text-halign": "center",
      color: "#e6edf3",
      "text-wrap": "wrap",
      "text-max-width": 120,
      "background-color": "#2b3a4d",
      "border-width": 1,
      "border-color": "#3b4d66",
      "text-outline-color": "#0b0f14",
      "text-outline-width": 2,
    },
  },
  {
    selector: 'node[type = "file"]',
    style: {
      shape: "round-rectangle",
      "background-color": "#284b63",
      "border-color": "#3b82c4",
      width: 120,
      height: 32,
      "font-weight": 600,
    },
  },
  {
    selector: 'node[type = "symbol"]',
    style: {
      shape: "ellipse",
      "background-color": "#1a3b2b",
      "border-color": "#2b8a5a",
      width: 16,
      height: 16,
    },
  },
  {
    selector: 'node[type = "open_promise"]',
    style: {
      shape: "diamond",
      "background-color": "#5b1c24",
      "border-color": "#c04048",
      width: 20,
      height: 20,
    },
  },
  {
    selector: "edge",
    style: {
      width: 1,
      "line-color": "#2b3a4d",
      "target-arrow-color": "#2b3a4d",
      "target-arrow-shape": "triangle",
      "curve-style": "bezier",
    },
  },
  {
    selector: 'edge[type = "defines"]',
    style: { "line-color": "#2b8a5a", "target-arrow-color": "#2b8a5a" },
  },
  {
    selector: 'edge[type = "imports"]',
    style: { "line-color": "#3b82c4", "target-arrow-color": "#3b82c4" },
  },
  {
    selector: 'edge[type = "promise_open"]',
    style: {
      "line-color": "#c04048",
      "target-arrow-color": "#c04048",
      "line-style": "dashed",
      width: 2,
    },
  },
  {
    selector: 'edge[type = "promise_resolved"]',
    style: {
      "line-color": "#2b8a5a",
      "target-arrow-color": "#2b8a5a",
      width: 2,
    },
  },
];

let cy = null;

function setStatus(msg) {
  document.getElementById("status").textContent = msg;
}

function renderDetail(obj) {
  document.getElementById("detail").textContent = obj
    ? JSON.stringify(obj, null, 2)
    : "";
}

function renderPromises(promises) {
  const ul = document.getElementById("promises");
  ul.innerHTML = "";
  const open = promises.filter((p) => p.resolved_at === null);
  if (!open.length) {
    ul.innerHTML =
      '<li class="muted">No open promises — every cross-file reference is satisfied.</li>';
    return;
  }
  for (const p of open) {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${p.expected_module}::${p.expected_name}</strong>
      <span class="promise-file">owed by ${p.source_file_path}</span>`;
    ul.appendChild(li);
  }
}

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
}

async function refreshProjects() {
  const projects = await fetchJSON("/projects");
  const sel = document.getElementById("project");
  const previous = sel.value;
  sel.innerHTML = "";
  if (!projects.length) {
    const opt = document.createElement("option");
    opt.textContent = "— no projects yet —";
    opt.disabled = true;
    opt.selected = true;
    sel.appendChild(opt);
    setStatus("ingest a project first: POST /ingest/directory");
    return;
  }
  for (const p of projects) {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = p.name;
    sel.appendChild(opt);
  }
  sel.value = projects.some((p) => p.id === previous) ? previous : projects[0].id;
  await refreshGraph();
}

async function refreshGraph() {
  const projectId = document.getElementById("project").value;
  if (!projectId) return;
  setStatus("loading…");
  try {
    const [graph, promises] = await Promise.all([
      fetchJSON(`/projects/${encodeURIComponent(projectId)}/graph`),
      fetchJSON(`/projects/${encodeURIComponent(projectId)}/promises`),
    ]);
    const elements = [...graph.nodes, ...graph.edges];
    if (cy === null) {
      cy = cytoscape({
        container: document.getElementById("graph"),
        elements,
        style: STYLE,
        layout: { name: "fcose", animate: false, nodeRepulsion: 4500 },
        wheelSensitivity: 0.15,
      });
      cy.on("tap", "node", (e) => renderDetail(e.target.data()));
      cy.on("tap", "edge", (e) => renderDetail(e.target.data()));
    } else {
      cy.elements().remove();
      cy.add(elements);
      cy.layout({ name: "fcose", animate: false, nodeRepulsion: 4500 }).run();
    }
    renderPromises(promises);
    setStatus(
      `${graph.nodes.length} nodes · ${graph.edges.length} edges · ${promises.filter((p) => p.resolved_at === null).length} open promises`,
    );
  } catch (err) {
    setStatus(`error: ${err.message}`);
  }
}

document.getElementById("refresh").addEventListener("click", refreshGraph);
document.getElementById("project").addEventListener("change", refreshGraph);

refreshProjects();
setInterval(refreshProjects, 10000);
