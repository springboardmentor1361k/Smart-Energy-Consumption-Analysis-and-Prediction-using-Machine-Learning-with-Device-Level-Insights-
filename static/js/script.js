let currentFrame = "H";

/* ================== DASHBOARD REFRESH ================== */
function refresh(timeframe) {
    currentFrame = timeframe;
    const device = document.getElementById("deviceSelect")?.value || "All";

    fetch("/update_dashboard", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ device: device, view: timeframe })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("lineGraph").src =
            "data:image/png;base64," + data.line_graph;

        document.getElementById("pieGraph").src =
            "data:image/png;base64," + data.pie_graph;

        document.getElementById("topDeviceGraph").src =
            "data:image/png;base64," + data.top_device_graph;

        document.getElementById("predVal").innerText =
            data.prediction + " kW";

        const list = document.getElementById("suggestionsList");
        list.innerHTML = "";
        data.suggestions.forEach(s => {
            const li = document.createElement("li");
            li.innerText = s;
            list.appendChild(li);
        });

        // ✅ Update device toggles
        updateDeviceControls(data.device_status);
        updateQuickControls(data.device_status);
    })
    .catch(err => console.error("Dashboard error:", err));
}

/* ================== DEVICE CONTROLS ================== */
function updateDeviceControls(devices) {
    const container = document.getElementById("deviceControls");
    if (!container) return;

    container.innerHTML = "";
    Object.keys(devices).forEach(device => {
        container.innerHTML += `
        <div class="device-row">
            <span class="device-name">${device}</span>
            <label class="switch">
                <input type="checkbox"
                    ${devices[device] ? "checked" : ""}
                    onchange="toggleDevice('${device}', this.checked)">
                <span class="slider"></span>
            </label>
        </div>`;
    });
}

function updateQuickControls(devices) {
    const container = document.getElementById("quickDeviceControls");
    if (!container) return;

    container.innerHTML = "";
    Object.keys(devices).forEach(device => {
        container.innerHTML += `
        <div class="device-row">
            <span class="device-name">${device}</span>
            <label class="switch">
                <input type="checkbox"
                    ${devices[device] ? "checked" : ""}
                    onchange="toggleDevice('${device}', this.checked)">
                <span class="slider"></span>
            </label>
        </div>`;
    });
}

function toggleDevice(device, state) {
    fetch("/toggle_device", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ device: device, state: state })
    })
    .then(() => refresh(currentFrame));
}

/* ================== MANUAL PREDICTION ================== */
function predictManual() {
    const inputStr = document.getElementById("manualInput").value;
    const resultLabel = document.getElementById("manualResult");

    if (!inputStr) return alert("Enter readings");

    const values = inputStr.split(",").map(v => Number(v.trim()));
    if (values.some(isNaN)) return alert("Invalid numbers");

    fetch("/predict_manual", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input: values })
    })
    .then(res => res.json())
    .then(data => {
        resultLabel.innerHTML =
            `Forecasted Next Reading: <span class="neon-blue">${data.prediction} kW</span>`;
    });
}

/* ================== CHATBOT ================== */
function sendChat() {
    const input = document.getElementById("chatInput");
    const content = document.getElementById("chat-messages");
    const msg = input.value.trim();
    if (!msg) return;

    // User message
    content.innerHTML += `<div><b>You:</b> ${msg}</div>`;
    content.scrollTop = content.scrollHeight;
    input.value = "";

    // Typing indicator
    const typingId = "typing-" + Date.now();
    content.innerHTML += `<div id="${typingId}" style="color:#888;">Assistant is typing...</div>`;
    content.scrollTop = content.scrollHeight;

    fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById(typingId)?.remove();

        content.innerHTML +=
            `<div style="color:#00e6e6;"><b>Assistant:</b> ${data.reply}</div>`;
        content.scrollTop = content.scrollHeight;

        if (data.refresh) refresh(currentFrame);
    })
    .catch(() => {
        document.getElementById(typingId)?.remove();
        content.innerHTML +=
            `<div style="color:red;"><b>Assistant:</b> Something went wrong.</div>`;
    });
}

/* Send chat on Enter key */
document.addEventListener("DOMContentLoaded", () => {
    const chatInput = document.getElementById("chatInput");
    if (chatInput) {
        chatInput.addEventListener("keypress", e => {
            if (e.key === "Enter") sendChat();
        });
    }
});

/* Toggle Chat Window */
function toggleChat() {
    const box = document.getElementById("chat-box");
    box.style.display =
        (box.style.display === "none" || box.style.display === "") ? "flex" : "none";
}

/* ================== DEFAULT LOAD ================== */
window.onload = () => refresh("H");
