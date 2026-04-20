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

function appendMessage(role, text, meta) {
  const box = document.getElementById("messages");
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  if (role === "assistant" && meta) {
    const m = document.createElement("div");
    m.className = "meta";
    m.textContent = meta;
    div.appendChild(m);
  }
  const body = document.createElement("div");
  body.className = "body";
  body.textContent = text;
  div.appendChild(body);
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return body;
}

function attachContextDetails(div, contextText) {
  if (!contextText) return;
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = `memory injected (${contextText.length} chars)`;
  const pre = document.createElement("pre");
  pre.textContent = contextText;
  details.appendChild(summary);
  details.appendChild(pre);
  div.appendChild(details);
}

async function sendChatStream(projectId, message, model) {
  appendMessage("user", message);
  const body = appendMessage("assistant", "…", `${model}`);
  const r = await fetch("/chat/stream", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ project_id: projectId, message, model }),
  });
  if (!r.ok) {
    body.textContent = `error: HTTP ${r.status}`;
    return;
  }
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  body.textContent = "";
  let contextText = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 1);
      if (!line.trim()) continue;
      try {
        const evt = JSON.parse(line);
        if (evt.type === "context") contextText = evt.context || "";
        else if (evt.type === "token") body.textContent += evt.text;
        else if (evt.type === "error") body.textContent += `\n[error: ${evt.detail}]`;
      } catch {}
    }
  }
  attachContextDetails(body.parentElement, contextText);
  setStatus("ready");
  refreshGraph();
}

async function sendChatSingleShot(projectId, message, model) {
  appendMessage("user", message);
  const body = appendMessage("assistant", "…", `${model}`);
  const r = await fetch("/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ project_id: projectId, message, model }),
  });
  if (!r.ok) {
    body.textContent = `error: HTTP ${r.status}`;
    return;
  }
  const data = await r.json();
  body.textContent = data.reply || "(empty response)";
  attachContextDetails(body.parentElement, data.context || "");
  setStatus("ready");
  refreshGraph();
}

async function refreshModels() {
  try {
    const data = await fetchJSON("/chat/models");
    const sel = document.getElementById("model");
    sel.innerHTML = "";
    for (const name of data.models) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      sel.appendChild(opt);
    }
    if (data.default && data.models.includes(data.default)) {
      sel.value = data.default;
    }
  } catch (err) {
    const sel = document.getElementById("model");
    sel.innerHTML = "";
    const opt = document.createElement("option");
    opt.textContent = "Ollama unavailable";
    opt.disabled = true;
    sel.appendChild(opt);
  }
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

function bindChatForm() {
  const form = document.getElementById("chat-form");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = document.getElementById("chat-input");
    const message = input.value.trim();
    if (!message) return;
    const projectId = document.getElementById("project").value || "default";
    const model = document.getElementById("model").value || undefined;
    const streaming = document.getElementById("stream-toggle").checked;
    input.value = "";
    setStatus("thinking…");
    try {
      if (streaming) await sendChatStream(projectId, message, model);
      else await sendChatSingleShot(projectId, message, model);
    } catch (err) {
      setStatus(`chat error: ${err.message}`);
    }
  });
  document.getElementById("chat-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      form.requestSubmit();
    }
  });
}

document.getElementById("refresh").addEventListener("click", refreshGraph);
document.getElementById("project").addEventListener("change", refreshGraph);

bindChatForm();
refreshModels();
refreshProjects();
setInterval(refreshProjects, 10000);
