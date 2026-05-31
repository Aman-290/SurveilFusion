const camerasEl = document.querySelector("#cameras");
const eventsEl = document.querySelector("#events");
const cameraCountEl = document.querySelector("#camera-count");
const eventCountEl = document.querySelector("#event-count");
const openCountEl = document.querySelector("#open-count");
const demoButton = document.querySelector("#demo-event");

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
    </article>
  `).join("") || "<p class=\"meta\">No incidents yet.</p>";
}

async function refresh() {
  const [cameras, events] = await Promise.all([
    fetchJson("/api/cameras"),
    fetchJson("/api/events"),
  ]);
  renderCameras(cameras);
  renderEvents(events);
}

demoButton.addEventListener("click", async () => {
  await fetchJson("/api/events/demo", { method: "POST" });
  await refresh();
});

refresh().catch((error) => {
  eventsEl.innerHTML = `<p class="critical">${error.message}</p>`;
});
