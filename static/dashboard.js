const modeButtons = document.querySelectorAll(".mode-btn");
const badge = document.getElementById("conn-badge");
const eventsBody = document.querySelector("#events-table tbody");
const cameraSelect = document.getElementById("camera-select");
const customCameraBox = document.getElementById("custom-camera-box");
const customCameraInput = document.getElementById("custom-camera-input");
const saveCustomCameraBtn = document.getElementById("save-custom-camera");
const deleteCustomCameraBtn = document.getElementById("delete-custom-camera");
const queueControls = document.getElementById("queue-roi-controls");
const clearRoiBtn = document.getElementById("clear-roi-btn");
const videoFeed = document.getElementById("video-feed");
const videoContainer = document.getElementById("video-container");
const roiSvg = document.getElementById("roi-svg");
const clearLogsBtn = document.getElementById("clear-logs-btn");

const modelDataHead = document.getElementById("model-data-head");
const modelDataBody = document.getElementById("model-data-body");

let roiPoints = [];
let currentMode = "access";

// --- Full Screen Logic ---
videoContainer.addEventListener("click", (e) => {
  if (e.target.tagName.toLowerCase() === 'svg' || e.target.tagName.toLowerCase() === 'circle' || e.target.tagName.toLowerCase() === 'polygon') return;
  
  if (!document.fullscreenElement) {
    videoContainer.requestFullscreen().catch(err => {
      alert(`Error attempting to enable fullscreen: ${err.message}`);
    });
  } else {
    document.exitFullscreen();
  }
});

// --- Camera Logic ---
function loadCustomCameras() {
    const saved = localStorage.getItem("custom_cameras");
    if (saved) {
        try {
            const cameras = JSON.parse(saved);
            cameras.forEach(cam => {
                const opt = document.createElement("option");
                opt.value = cam;
                opt.textContent = `Custom: ${cam}`;
                opt.dataset.custom = "true";
                // Insert before 'custom' option
                cameraSelect.insertBefore(opt, cameraSelect.querySelector("option[value='custom']"));
            });
        } catch (e) {}
    }
}
loadCustomCameras();

cameraSelect.addEventListener("change", async () => {
    if (cameraSelect.value === "custom") {
        customCameraBox.style.display = "flex";
        deleteCustomCameraBtn.style.display = "none";
    } else {
        customCameraBox.style.display = "none";
        const isCustom = cameraSelect.options[cameraSelect.selectedIndex].dataset.custom === "true";
        deleteCustomCameraBtn.style.display = isCustom ? "inline-block" : "none";
        
        await fetch("/api/set_camera", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: cameraSelect.value }),
        });
        videoFeed.src = "/video_feed?_=" + Date.now();
    }
});

saveCustomCameraBtn.addEventListener("click", async () => {
    const url = customCameraInput.value.trim();
    if (!url) return;
    
    // Save to local storage
    const saved = localStorage.getItem("custom_cameras");
    let cameras = saved ? JSON.parse(saved) : [];
    if (!cameras.includes(url)) {
        cameras.push(url);
        localStorage.setItem("custom_cameras", JSON.stringify(cameras));
        
        const opt = document.createElement("option");
        opt.value = url;
        opt.textContent = `Custom: ${url}`;
        opt.dataset.custom = "true";
        cameraSelect.insertBefore(opt, cameraSelect.querySelector("option[value='custom']"));
    }
    
    cameraSelect.value = url;
    customCameraBox.style.display = "none";
    deleteCustomCameraBtn.style.display = "inline-block";
    customCameraInput.value = "";
    
    await fetch("/api/set_camera", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
    });
    videoFeed.src = "/video_feed?_=" + Date.now();
});

deleteCustomCameraBtn.addEventListener("click", () => {
    const url = cameraSelect.value;
    const saved = localStorage.getItem("custom_cameras");
    if (saved) {
        let cameras = JSON.parse(saved);
        cameras = cameras.filter(c => c !== url);
        localStorage.setItem("custom_cameras", JSON.stringify(cameras));
    }
    cameraSelect.options[cameraSelect.selectedIndex].remove();
    cameraSelect.selectedIndex = 0;
    deleteCustomCameraBtn.style.display = "none";
});

// --- ROI SVG ---
function drawRoiSvg() {
  if (!roiSvg) return;
  roiSvg.innerHTML = "";
  
  if (roiPoints.length === 0) return;
  
  const rect = roiSvg.getBoundingClientRect();
  
  for (let i = 0; i < roiPoints.length; i += 4) {
    const chunk = roiPoints.slice(i, i + 4);
    const queueIndex = Math.floor(i / 4) + 1;
    
    chunk.forEach((pt, index) => {
      const cx = pt[0] * rect.width;
      const cy = pt[1] * rect.height;
      
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", cx);
      circle.setAttribute("cy", cy);
      circle.setAttribute("r", "6");
      circle.setAttribute("fill", "#00ff00");
      circle.setAttribute("stroke", "#fff");
      circle.setAttribute("stroke-width", "2");
      roiSvg.appendChild(circle);
      
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", cx + 8);
      text.setAttribute("y", cy + 5);
      text.setAttribute("fill", "#fff");
      text.setAttribute("font-size", "12");
      text.setAttribute("font-weight", "bold");
      text.setAttribute("paint-order", "stroke");
      text.setAttribute("stroke", "#000");
      text.setAttribute("stroke-width", "3");
      text.textContent = `Q${queueIndex}.${index + 1}`;
      roiSvg.appendChild(text);
    });
    
    if (chunk.length > 1) {
      const pointsStr = chunk.map(pt => `${pt[0] * rect.width},${pt[1] * rect.height}`).join(" ");
      if (chunk.length === 4) {
        const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
        polygon.setAttribute("points", pointsStr);
        polygon.setAttribute("fill", "rgba(0, 255, 0, 0.15)");
        polygon.setAttribute("stroke", "#00ff00");
        polygon.setAttribute("stroke-width", "2");
        roiSvg.appendChild(polygon);
      } else {
        const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
        polyline.setAttribute("points", pointsStr);
        polyline.setAttribute("fill", "none");
        polyline.setAttribute("stroke", "#00ff00");
        polyline.setAttribute("stroke-width", "2");
        polyline.setAttribute("stroke-dasharray", "4");
        roiSvg.appendChild(polyline);
      }
    }
  }
}

window.addEventListener("resize", drawRoiSvg);

function updateControlsVisibility(mode) {
  if (queueControls) queueControls.style.display = (mode === "queue") ? "block" : "none";
  if (boxLineControls) boxLineControls.style.display = (mode === "box") ? "block" : "none";
  if (peopleLineControls) peopleLineControls.style.display = (mode === "people") ? "block" : "none";
  
  if (roiSvg) {
      roiSvg.style.display = (mode === "queue" || mode === "box" || mode === "people") ? "block" : "none";
      if (mode === "queue") drawRoiSvg();
      else if (mode === "box") drawBoxLineSvg();
      else if (mode === "people") drawPeopleLineSvg();
  }
}

function highlightActiveMode(mode) {
  currentMode = mode;
  modeButtons.forEach(btn => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });
  updateControlsVisibility(mode);
}

modeButtons.forEach(btn => {
  btn.addEventListener("click", async () => {
    const mode = btn.dataset.mode;
    const res = await fetch("/api/set_mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    const data = await res.json();
    if (data.ok) {
        highlightActiveMode(mode);
        pollModelData(); // refresh table immediately
    }
  });
});

if (roiSvg) {
  roiSvg.addEventListener("click", async (e) => {
    const activeBtn = document.querySelector(".mode-btn.active");
    if (!activeBtn || activeBtn.dataset.mode !== "queue") return;

    const rect = roiSvg.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    
    const relX = clickX / rect.width;
    const relY = clickY / rect.height;
    
    roiPoints.push([relX, relY]);
    drawRoiSvg();
    
    const selectEl = document.getElementById("num-queues-select");
    const numQueues = selectEl ? parseInt(selectEl.value) : 1;
    const maxPoints = numQueues * 4;

    if (roiPoints.length === maxPoints) {
      let chunks = [];
      for (let i = 0; i < maxPoints; i += 4) {
        chunks.push(roiPoints.slice(i, i + 4));
      }
      const res = await fetch("/api/set_queue_roi", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ points: chunks }),
      });
      const data = await res.json();
      if (data.ok) {
        setTimeout(() => {
          roiPoints = [];
          drawRoiSvg();
        }, 800);
      } else {
        alert("Error setting ROI: " + data.error);
        roiPoints = [];
        drawRoiSvg();
      }
    }
  });
}

clearRoiBtn.addEventListener("click", async () => {
  roiPoints = [];
  drawRoiSvg();
  
  const res = await fetch("/api/set_queue_roi", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ points: [] }),
  });
  const data = await res.json();
  if (!data.ok) {
    alert("Error resetting Queue Zone: " + data.error);
  }
});

// --- Box Counter SVG Logic ---
const boxLineControls = document.getElementById("box-line-controls");
const clearLineBtn = document.getElementById("clear-line-btn");
let boxPoints = [];

function drawBoxLineSvg() {
  if (!roiSvg) return;
  roiSvg.innerHTML = "";
  
  if (boxPoints.length === 0) return;
  
  const rect = roiSvg.getBoundingClientRect();
  
  boxPoints.forEach((pt, index) => {
    const cx = pt[0] * rect.width;
    const cy = pt[1] * rect.height;
    
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", cx);
    circle.setAttribute("cy", cy);
    circle.setAttribute("r", "6");
    circle.setAttribute("fill", "#3498db");
    circle.setAttribute("stroke", "#fff");
    circle.setAttribute("stroke-width", "2");
    roiSvg.appendChild(circle);
  });
  
  if (boxPoints.length === 2) {
    const p1 = boxPoints[0];
    const p2 = boxPoints[1];
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", p1[0] * rect.width);
    line.setAttribute("y1", p1[1] * rect.height);
    line.setAttribute("x2", p2[0] * rect.width);
    line.setAttribute("y2", p2[1] * rect.height);
    line.setAttribute("stroke", "#3498db");
    line.setAttribute("stroke-width", "3");
    roiSvg.appendChild(line);
  }
}

if (roiSvg) {
  roiSvg.addEventListener("click", async (e) => {
    const activeBtn = document.querySelector(".mode-btn.active");
    if (!activeBtn || activeBtn.dataset.mode !== "box") return;

    if (boxPoints.length >= 2) return; // Already have a line

    const rect = roiSvg.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    
    const relX = clickX / rect.width;
    const relY = clickY / rect.height;
    
    boxPoints.push([relX, relY]);
    drawBoxLineSvg();
    
    if (boxPoints.length === 2) {
      const res = await fetch("/api/set_box_line", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ points: boxPoints }),
      });
      const data = await res.json();
      if (!data.ok) {
        alert("Error setting line: " + data.error);
        boxPoints = [];
        drawBoxLineSvg();
      }
    }
  });
}

if (clearLineBtn) {
  clearLineBtn.addEventListener("click", async () => {
    boxPoints = [];
    drawBoxLineSvg();
    
    await fetch("/api/set_box_line", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ points: [] }),
    });
  });
}

// --- People Counter SVG Logic ---
const peopleLineControls = document.getElementById("people-line-controls");
const clearPeopleLineBtn = document.getElementById("clear-people-line-btn");
let peoplePoints = [];

function drawPeopleLineSvg() {
  if (!roiSvg) return;
  roiSvg.innerHTML = "";
  
  if (peoplePoints.length === 0) return;
  
  const rect = roiSvg.getBoundingClientRect();
  
  peoplePoints.forEach((pt, index) => {
    const cx = pt[0] * rect.width;
    const cy = pt[1] * rect.height;
    
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", cx);
    circle.setAttribute("cy", cy);
    circle.setAttribute("r", "6");
    circle.setAttribute("fill", index < 2 ? "#00ff00" : "#00ffff");
    circle.setAttribute("stroke", "#fff");
    circle.setAttribute("stroke-width", "2");
    roiSvg.appendChild(circle);
  });
  
  // Draw first line (outside)
  if (peoplePoints.length >= 2) {
    const p1 = peoplePoints[0];
    const p2 = peoplePoints[1];
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", p1[0] * rect.width);
    line.setAttribute("y1", p1[1] * rect.height);
    line.setAttribute("x2", p2[0] * rect.width);
    line.setAttribute("y2", p2[1] * rect.height);
    line.setAttribute("stroke", "#00ff00");
    line.setAttribute("stroke-width", "3");
    roiSvg.appendChild(line);
  }

  // Draw second line (inside)
  if (peoplePoints.length === 4) {
    const p1 = peoplePoints[2];
    const p2 = peoplePoints[3];
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", p1[0] * rect.width);
    line.setAttribute("y1", p1[1] * rect.height);
    line.setAttribute("x2", p2[0] * rect.width);
    line.setAttribute("y2", p2[1] * rect.height);
    line.setAttribute("stroke", "#00ffff");
    line.setAttribute("stroke-width", "3");
    roiSvg.appendChild(line);
  }
}

if (roiSvg) {
  roiSvg.addEventListener("click", async (e) => {
    const activeBtn = document.querySelector(".mode-btn.active");
    if (!activeBtn || activeBtn.dataset.mode !== "people") return;

    if (peoplePoints.length >= 4) return; // Already have 2 lines

    const rect = roiSvg.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    
    const relX = clickX / rect.width;
    const relY = clickY / rect.height;
    
    peoplePoints.push([relX, relY]);
    drawPeopleLineSvg();
    
    if (peoplePoints.length === 4) {
      let chunks = [
          [peoplePoints[0][0], peoplePoints[0][1], peoplePoints[1][0], peoplePoints[1][1]],
          [peoplePoints[2][0], peoplePoints[2][1], peoplePoints[3][0], peoplePoints[3][1]]
      ];
      const res = await fetch("/api/set_people_lines", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ points: chunks }),
      });
      const data = await res.json();
      if (!data.ok) {
        alert("Error setting lines: " + data.error);
        peoplePoints = [];
        drawPeopleLineSvg();
      }
    }
  });
}

if (clearPeopleLineBtn) {
  clearPeopleLineBtn.addEventListener("click", async () => {
    peoplePoints = [];
    drawPeopleLineSvg();
    
    await fetch("/api/set_people_lines", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ points: [] }),
    });
  });
}

const resetPeopleCountBtn = document.getElementById("reset-people-count-btn");
if (resetPeopleCountBtn) {
  resetPeopleCountBtn.addEventListener("click", async () => {
    await fetch("/api/reset_people_counts", { method: "POST" });
  });
}


// Ensure resizing redraws correct SVG
window.addEventListener("resize", () => {
    if (currentMode === "queue") drawRoiSvg();
    if (currentMode === "box") drawBoxLineSvg();
    if (currentMode === "people") drawPeopleLineSvg();
});

// --- Clear Logs ---
clearLogsBtn.addEventListener("click", async () => {
    await fetch("/api/clear_events", { method: "POST" });
    pollEvents();
});

// --- API Polling ---
async function pollStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    badge.textContent = data.connected ? "connected" : (data.error || "connecting...");
    badge.className = "badge " + (data.connected ? "connected" : "disconnected");
    if (data.mode !== currentMode) {
        highlightActiveMode(data.mode);
        pollModelData();
    }
    
    // Sync dropdown if it matches a known camera but don't overwrite custom unsaved text
    if (cameraSelect.value !== "custom" && data.camera_url) {
        let found = false;
        for (let i = 0; i < cameraSelect.options.length; i++) {
            if (cameraSelect.options[i].value === data.camera_url) {
                cameraSelect.selectedIndex = i;
                found = true;
                break;
            }
        }
        if (!found) {
            // It's a new unknown url not in local storage yet (from backend load)
            const opt = document.createElement("option");
            opt.value = data.camera_url;
            opt.textContent = `Custom: ${data.camera_url}`;
            opt.dataset.custom = "true";
            cameraSelect.insertBefore(opt, cameraSelect.querySelector("option[value='custom']"));
            cameraSelect.value = data.camera_url;
        }
    }
  } catch (e) { /* ignore transient errors */ }
}

function statusClass(status) {
  return "status-" + (status || "info");
}

async function pollEvents() {
  try {
    const res = await fetch("/api/events");
    const events = await res.json();
    eventsBody.innerHTML = events.map(ev => {
      const t = new Date(ev.ts * 1000).toLocaleTimeString();
      return `<tr>
        <td>${t}</td>
        <td>${ev.mode}</td>
        <td>${ev.event_type}</td>
        <td>${ev.detail || ""}</td>
        <td class="${statusClass(ev.status)}">${ev.status || ""}</td>
      </tr>`;
    }).join("");
  } catch (e) { /* ignore transient errors */ }
}

async function pollModelData() {
    try {
        const res = await fetch(`/api/model_data?mode=${currentMode}`);
        const data = await res.json();
        
        let headers = "";
        let rows = "";
        let title = "Log Table";
        if (currentMode === "access") {
            title = "Access Control Log Table";
            headers = "<th>Serial No.</th><th>Name</th><th>Access Result</th><th>Photo</th><th>Date</th><th>Time</th>";
            rows = data.map(r => {
                const dt = new Date(r.ts * 1000);
                const dateStr = dt.toISOString().split('T')[0];
                const timeStr = dt.toTimeString().split(' ')[0];
                const imgHtml = r.photo_path ? `<img src="/${r.photo_path}" style="width: 80px; height: 45px; object-fit: cover; border-radius: 4px; border: 1px solid #444;">` : 'N/A';
                // Show Alert if denied, else name
                const displayObj = r.status === 'denied' ? 'Alert' : r.name;
                return `<tr>
                    <td>${r.id}</td>
                    <td>${displayObj}</td>
                    <td class="${statusClass(r.status === 'granted' ? 'success' : 'error')}">${r.status}</td>
                    <td>${imgHtml}</td>
                    <td>${dateStr}</td>
                    <td>${timeStr}</td>
                </tr>`;
            }).join("");
        } else if (currentMode === "adaptive") {
            title = "Adaptive Bitrate Log Table";
            headers = "<th>Serial No.</th><th>Bitrate State</th><th>Duration (s)</th><th>Photo</th><th>Date</th><th>Time</th>";
            rows = data.map(r => {
                const dt = new Date(r.ts * 1000);
                const dateStr = dt.toISOString().split('T')[0];
                const timeStr = dt.toTimeString().split(' ')[0];
                const imgHtml = r.photo_path ? `<img src="/${r.photo_path}" style="width: 80px; height: 45px; object-fit: cover; border-radius: 4px; border: 1px solid #444;">` : 'N/A';
                return `<tr>
                    <td>${r.id}</td>
                    <td>${r.state.toUpperCase()}</td>
                    <td>${r.duration.toFixed(1)}</td>
                    <td>${imgHtml}</td>
                    <td>${dateStr}</td>
                    <td>${timeStr}</td>
                </tr>`;
            }).join("");
        } else if (currentMode === "people") {
            title = "People Counter Log Table";
            headers = "<th>Serial No.</th><th>Direction</th><th>Total Entries</th><th>Total Exits</th><th>Photo</th><th>Date</th><th>Time</th>";
            rows = data.map(r => {
                const dt = new Date(r.ts * 1000);
                const dateStr = dt.toISOString().split('T')[0];
                const timeStr = dt.toTimeString().split(' ')[0];
                const imgHtml = r.photo_path ? `<img src="/${r.photo_path}" style="width: 80px; height: 45px; object-fit: cover; border-radius: 4px; border: 1px solid #444;">` : 'N/A';
                return `<tr>
                    <td>${r.id}</td>
                    <td>${r.direction.toUpperCase()}</td>
                    <td>${r.total_entries}</td>
                    <td>${r.total_exits}</td>
                    <td>${imgHtml}</td>
                    <td>${dateStr}</td>
                    <td>${timeStr}</td>
                </tr>`;
            }).join("");
        } else if (currentMode === "queue") {
            title = "Queue Monitor Log Table";
            headers = "<th>Serial No.</th><th>Person Count</th><th>Photo</th><th>Date</th><th>Time</th>";
            rows = data.map(r => {
                const dt = new Date(r.ts * 1000);
                const dateStr = dt.toISOString().split('T')[0];
                const timeStr = dt.toTimeString().split(' ')[0];
                const imgHtml = r.photo_path ? `<img src="/${r.photo_path}" style="width: 80px; height: 45px; object-fit: cover; border-radius: 4px; border: 1px solid #444;">` : 'N/A';
                return `<tr>
                    <td>${r.id}</td>
                    <td>${r.person_count}</td>
                    <td>${imgHtml}</td>
                    <td>${dateStr}</td>
                    <td>${timeStr}</td>
                </tr>`;
            }).join("");
        } else if (currentMode === "worker") {
            title = "Worker Tracker Log Table";
            headers = "<th>Serial No.</th><th>Worker Name</th><th>Work Time (s)</th><th>Rest Time (s)</th><th>Photo</th><th>Date</th><th>Time</th>";
            rows = data.map(r => {
                const dt = new Date(r.ts * 1000);
                const dateStr = dt.toISOString().split('T')[0];
                const timeStr = dt.toTimeString().split(' ')[0];
                const imgHtml = r.photo_path ? `<img src="/${r.photo_path}" style="width: 80px; height: 45px; object-fit: cover; border-radius: 4px; border: 1px solid #444;">` : 'N/A';
                return `<tr>
                    <td>${r.id}</td>
                    <td>${r.worker_name}</td>
                    <td>${r.work_time_s.toFixed(1)}</td>
                    <td>${r.rest_time_s.toFixed(1)}</td>
                    <td>${imgHtml}</td>
                    <td>${dateStr}</td>
                    <td>${timeStr}</td>
                </tr>`;
            }).join("");
        } else if (currentMode === "box") {
            title = "Bags & Boxes Counter";
            headers = "<th>Serial No.</th><th>Boxes IN</th><th>Boxes OUT</th><th>Photo</th><th>Date</th><th>Time</th>";
            rows = data.map(r => {
                const dt = new Date(r.ts * 1000);
                const dateStr = dt.toISOString().split('T')[0];
                const timeStr = dt.toTimeString().split(' ')[0];
                const imgHtml = r.photo_path ? `<img src="/${r.photo_path}" style="width: 80px; height: 45px; object-fit: cover; border-radius: 4px; border: 1px solid #444;">` : 'N/A';
                return `<tr>
                    <td>${r.id}</td>
                    <td><strong style="color: #2ecc71;">${r.boxes_in}</strong></td>
                    <td><strong style="color: #e74c3c;">${r.boxes_out}</strong></td>
                    <td>${imgHtml}</td>
                    <td>${dateStr}</td>
                    <td>${timeStr}</td>
                </tr>`;
            }).join("");
        } else {
            headers = "<th>Serial No.</th><th>Time</th><th>Info</th>";
            rows = "<tr><td colspan='3'>No structured data available for this mode.</td></tr>";
        }
        
        // Update table title if an element exists
        const tableTitle = document.getElementById("model-data-title");
        if (tableTitle) tableTitle.textContent = title;
        
        modelDataHead.innerHTML = headers;
        modelDataBody.innerHTML = rows;
        
    } catch (e) { console.error("Error polling model data:", e); }
}


pollStatus();
pollEvents();
pollModelData();
setInterval(pollStatus, 2000);
setInterval(pollEvents, 2000);
setInterval(pollModelData, 5000); // Live details updated every 5s
