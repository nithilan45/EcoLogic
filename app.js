const promptInput = document.getElementById("prompt");
const sendBtn = document.getElementById("send");
const welcomeEl = document.getElementById("chat-welcome");
const responseEl = document.getElementById("chat-response");
const responseTextEl = document.getElementById("response-text");
const tierEl = document.getElementById("tier");
const energyUsedEl = document.getElementById("energy-used");
const energySavedEl = document.getElementById("energy-saved");
const energySavedGpt5El = document.getElementById("energy-saved-gpt5");

const API_URL = "https://ecologic-production.up.railway.app";

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
  responseTextEl.innerHTML = '<span class="typing-cursor"></span>';
  
  // Reset energy stats
  tierEl.textContent = "-";
  energyUsedEl.textContent = "-";
  energySavedEl.textContent = "-";
  energySavedGpt5El.textContent = "-";

  let fullContent = "";

  try {
    const response = await fetch(`${API_URL}/query/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ prompt: text }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split("\n");

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6));
            console.log('Received data type:', data.type, 'Full data:', JSON.stringify(data)); // Debug log
            
            if (data.type === "meta") {
              tierEl.textContent = data.tier;
            } else if (data.type === "content") {
              fullContent += data.content;
              responseTextEl.innerHTML = marked.parse(fullContent) + '<span class="typing-cursor"></span>';
            } else if (data.type === "done") {
              console.log('✅ Done event received! Energy:', data.energy_used, 'Saved:', data.energy_saved); // Debug log
              energyUsedEl.textContent = formatEnergy(data.energy_used);
              energySavedEl.textContent = formatEnergy(data.energy_saved);
              energySavedGpt5El.textContent = formatEnergy(data.energy_saved_vs_gpt5);
            }
          } catch (e) {
            console.error('❌ JSON parse error:', e, 'Line:', line); // Better error logging
          }
        }
      }
    }

    // Final render without cursor
    responseTextEl.innerHTML = marked.parse(fullContent);
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
