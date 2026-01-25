const promptInput = document.getElementById("prompt");
const sendBtn = document.getElementById("send");
const welcomeEl = document.getElementById("chat-welcome");
const responseEl = document.getElementById("chat-response");
const responseTextEl = document.getElementById("response-text");
const tierEl = document.getElementById("tier");
const energyUsedEl = document.getElementById("energy-used");
const energySavedEl = document.getElementById("energy-saved");

const API_URL = "http://localhost:8080";

const formatEnergy = (value) => `${value.toFixed(1)} J`;

// Configure marked for clean output
marked.setOptions({
  breaks: true,
  gfm: true,
});

const setLoading = (loading) => {
  sendBtn.disabled = loading;
  sendBtn.style.opacity = loading ? "0.5" : "1";
  promptInput.disabled = loading;
};

const handleSend = async () => {
  const text = promptInput.value.trim();
  if (!text) return;

  setLoading(true);
  welcomeEl.setAttribute("hidden", "");
  responseEl.removeAttribute("hidden");
  responseTextEl.textContent = "Thinking...";

  try {
    const response = await fetch(`${API_URL}/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ prompt: text }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();

    responseTextEl.innerHTML = marked.parse(data.response);
    tierEl.textContent = data.tier;
    energyUsedEl.textContent = formatEnergy(data.energy_used);
    energySavedEl.textContent = formatEnergy(data.energy_saved);
  } catch (error) {
    console.error("Error:", error);
    responseTextEl.textContent = `Error: ${error.message}. Make sure the backend is running on ${API_URL}`;
  } finally {
    setLoading(false);
    promptInput.value = "";
  }
};

sendBtn.addEventListener("click", handleSend);
promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    handleSend();
  }
});
