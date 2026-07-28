const nameInput = document.getElementById("enroll-name");
const fileInput = document.getElementById("photo-file");
const uploadBtn = document.getElementById("upload-btn");
const captureBtn = document.getElementById("capture-btn");
const statusMsg = document.getElementById("enroll-status");

function showStatus(text, ok) {
  statusMsg.textContent = text;
  statusMsg.style.color = ok ? "#7CFC7C" : "#ff7c7c";
}

uploadBtn.addEventListener("click", async () => {
  const name = nameInput.value.trim();
  const file = fileInput.files[0];
  if (!name || !file) {
    showStatus("Please enter a name and choose a photo.", false);
    return;
  }
  const formData = new FormData();
  formData.append("name", name);
  formData.append("photo", file);

  const res = await fetch("/api/enroll/upload", { method: "POST", body: formData });
  const data = await res.json();
  if (data.ok) {
    showStatus(`Saved ${data.filename}. Face recognition updated.`, true);
    setTimeout(() => location.reload(), 1200);
  } else {
    showStatus(data.error || "Upload failed.", false);
  }
});

captureBtn.addEventListener("click", async () => {
  const name = nameInput.value.trim();
  if (!name) {
    showStatus("Please enter a name first.", false);
    return;
  }
  const res = await fetch("/api/enroll/capture", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  const data = await res.json();
  if (data.ok) {
    showStatus(`Captured and saved ${data.filename}. Face recognition updated.`, true);
    setTimeout(() => location.reload(), 1200);
  } else {
    showStatus(data.error || "Capture failed.", false);
  }
});
