/* =====================================================
   GLOBAL STATE
===================================================== */
const AppState = {
    currentFrame: "H",
    devices: {}
};

/* =====================================================
   UTILITY HELPERS
===================================================== */
async function postJSON(url, payload) {
    const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error("Server error");
    return res.json();
}

function setImage(id, base64) {
    const el = document.getElementById(id);
    if (el && base64) {
        el.src = `data:image/png;base64,${base64}`;
    }
}

function clearElement(id) {
    const el = document.getElementById(id);
    if (el) el.innerHTML = "";
}

/* =====================================================
   DASHBOARD LOAD / REFRESH
===================================================== */
async function refreshDashboard(frame = "H") {
    try {
        AppState.currentFrame = frame;

        const device =
            document.getElementById("deviceSelect")?.value || "All";

        const data = await postJSON("/update_dashboard", {
            device,
            view: frame
        });

        updateGraphs(data);
        updatePrediction(data.prediction);
        updateSuggestions(data.suggestions);
        renderDevices(data.device_status);

    } catch (err) {
        console.error("Dashboard failed:", err);
    }
}

function updateGraphs(data) {
    setImage("lineGraph", data.line_graph);
    setImage("pieGraph", data.pie_graph);
    setImage("topDeviceGraph", data.top_device_graph);
}

function updatePrediction(value) {
    const el = document.getElementById("predVal");
    if (el) el.textContent = `${value} kW`;
}

function updateSuggestions(suggestions = []) {
    const list = document.getElementById("suggestionsList");
    if (!list) return;

    list.innerHTML = "";

    suggestions.forEach(text => {
        const li = document.createElement("li");
        li.textContent = text;
        list.appendChild(li);
    });
}

/* =====================================================
   DEVICE CONTROLS (Single Renderer)
===================================================== */
function renderDevices(devices) {
    AppState.devices = devices;

    renderDeviceGroup("deviceControls", devices);
    renderDeviceGroup("quickDeviceControls", devices);
}

function renderDeviceGroup(containerId, devices) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = "";

    Object.entries(devices).forEach(([name, state]) => {
        const row = document.createElement("div");
        row.className = "device-row";

        row.innerHTML = `
            <span class="device-name">${name}</span>
            <label class="switch">
                <input type="checkbox" data-device="${name}" ${state ? "checked" : ""}>
                <span class="slider"></span>
            </label>
        `;

        container.appendChild(row);
    });
}

/* =====================================================
   DEVICE TOGGLE (Event Delegation)
===================================================== */
document.addEventListener("change", async (e) => {
    if (!e.target.matches(".switch input")) return;

    const device = e.target.dataset.device;
    const state = e.target.checked;

    try {
        await postJSON("/toggle_device", { device, state });
        await refreshDashboard(AppState.currentFrame);
    } catch (err) {
        console.error("Toggle failed:", err);
    }
});

/* =====================================================
   MANUAL PREDICTION
===================================================== */
async function predictManual() {
    const input = document.getElementById("manualInput");
    const result = document.getElementById("manualResult");

    if (!input?.value) {
        alert("Please enter comma-separated readings.");
        return;
    }

    const values = input.value
        .split(",")
        .map(v => Number(v.trim()));

    if (values.some(isNaN)) {
        alert("Invalid numeric values.");
        return;
    }

    try {
        const data = await postJSON("/predict_manual", {
            input: values
        });

        result.innerHTML =
            `Forecasted Next Reading:
             <span class="neon-blue">${data.prediction} kW</span>`;
    } catch {
        result.textContent = "Prediction failed.";
    }
}

/* =====================================================
   CHAT SYSTEM
===================================================== */
async function sendChat() {
    const input = document.getElementById("chatInput");
    const box = document.getElementById("chat-messages");

    const message = input?.value.trim();
    if (!message) return;

    appendMessage("You", message, false);
    input.value = "";

    const typing = appendMessage("Assistant", "Typing...", true);

    try {
        const data = await postJSON("/chat", { message });

        typing.remove();
        appendMessage("Assistant", data.reply, true);

        if (data.refresh) {
            await refreshDashboard(AppState.currentFrame);
        }

    } catch {
        typing.remove();
        appendMessage("Assistant", "Something went wrong.", true);
    }
}

function appendMessage(sender, text, isAssistant) {
    const box = document.getElementById("chat-messages");
    if (!box) return;

    const div = document.createElement("div");
    div.className = isAssistant ? "chat-assistant" : "chat-user";
    div.innerHTML = `<strong>${sender}:</strong> ${text}`;

    box.appendChild(div);
    box.scrollTop = box.scrollHeight;

    return div;
}

/* =====================================================
   CHAT INPUT ENTER KEY
===================================================== */
document.addEventListener("DOMContentLoaded", () => {
    const input = document.getElementById("chatInput");

    input?.addEventListener("keydown", (e) => {
        if (e.key === "Enter") sendChat();
    });

    refreshDashboard("H");
});

/* =====================================================
   CHAT WINDOW TOGGLE
===================================================== */
function toggleChat() {
    const box = document.getElementById("chat-box");
    if (!box) return;

    box.classList.toggle("active");
}