const camerasEl = document.querySelector("#cameras");
const eventsEl = document.querySelector("#events");
const actionsEl = document.querySelector("#actions");
const memoryResultsEl = document.querySelector("#memory-results");
const cameraCountEl = document.querySelector("#camera-count");
const eventCountEl = document.querySelector("#event-count");
const openCountEl = document.querySelector("#open-count");
const actionCountEl = document.querySelector("#action-count");
const demoButton = document.querySelector("#demo-event");
const memoryForm = document.querySelector("#memory-form");
const memoryQuery = document.querySelector("#memory-query");

async function fetchJson(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(`Request failed: ${response.status}`);
  return response.json();
}

function renderCameras(cameras) {
  cameraCountEl.textContent = cameras.length;
  camerasEl.innerHTML = cameras.map(({ camera, status }) => `
    <article class="camera-card">
      <strong>${camera.name}</strong>
      <div class="meta">
        <span class="pill">${status}</span>
        <span class="pill">${camera.zone}</span>
        <span class="pill">${camera.detect_fps} FPS detect</span>
      </div>
    </article>
  `).join("");
}

function renderEvents(events) {
  eventCountEl.textContent = events.length;
  openCountEl.textContent = events.filter((event) => !event.acknowledged).length;
  eventsEl.innerHTML = events.map((event) => `
    <article class="event">
      <strong class="${event.severity}">${event.title}</strong>
      <p>${event.summary}</p>
      <div class="meta">
        <span class="pill">${event.camera_id}</span>
        <span class="pill">${event.kind}</span>
        <span class="pill">${new Date(event.created_at).toLocaleString()}</span>
      </div>
      <button class="inline-action" data-propose="${event.id}" type="button">Propose actions</button>
    </article>
  `).join("") || "<p class=\"meta\">No incidents yet.</p>";
}

function renderActions(actions) {
  actionCountEl.textContent = actions.length;
  actionsEl.innerHTML = actions.map((action) => `
    <article class="event">
      <strong class="${action.risk === "high" ? "critical" : ""}">${action.kind.replaceAll("_", " ")}</strong>
      <p>${action.reason}</p>
      <div class="meta">
        <span class="pill">${action.camera_id}</span>
        <span class="pill">${action.status}</span>
        <span class="pill">${action.risk} risk</span>
        <span class="pill">${action.requires_approval ? "approval required" : "auto"}</span>
      </div>
      <div class="button-row">
        <button class="inline-action" data-approve="${action.id}" type="button">Approve</button>
        <button class="inline-action" data-execute="${action.id}" type="button">Execute</button>
      </div>
    </article>
  `).join("") || "<p class=\"meta\">No actions queued.</p>";
}

function renderMemoryResults(results) {
  memoryResultsEl.innerHTML = results.map((result) => `
    <article class="event">
      <strong class="${result.event.severity}">${result.event.title}</strong>
      <p>${result.event.summary}</p>
      <div class="meta">
        <span class="pill">score ${result.score}</span>
        <span class="pill">${result.event.camera_id}</span>
        <span class="pill">${result.matched_terms.join(", ")}</span>
      </div>
    </article>
  `).join("") || "<p class=\"meta\">No matching incidents.</p>";
}

async function refresh() {
  const [cameras, events, actions] = await Promise.all([
    fetchJson("/api/cameras"),
    fetchJson("/api/events"),
    fetchJson("/api/actions"),
  ]);
  renderCameras(cameras);
  renderEvents(events);
  renderActions(actions);
}

demoButton.addEventListener("click", async () => {
  await fetchJson("/api/events/demo", { method: "POST" });
  await refresh();
});

memoryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = memoryQuery.value.trim();
  if (!query) {
    renderMemoryResults([]);
    return;
  }
  const results = await fetchJson(`/api/memory/search?q=${encodeURIComponent(query)}`);
  renderMemoryResults(results);
});

document.addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof HTMLButtonElement)) return;

  const proposeId = target.dataset.propose;
  const approveId = target.dataset.approve;
  const executeId = target.dataset.execute;

  if (proposeId) await fetchJson(`/api/events/${proposeId}/actions/propose`, { method: "POST" });
  if (approveId) await fetchJson(`/api/actions/${approveId}/approve`, { method: "POST" });
  if (executeId) await fetchJson(`/api/actions/${executeId}/execute`, { method: "POST" });
  if (proposeId || approveId || executeId) await refresh();
});

refresh().catch((error) => {
  eventsEl.innerHTML = `<p class="critical">${error.message}</p>`;
});
