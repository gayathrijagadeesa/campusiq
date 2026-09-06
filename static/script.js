const FEATURES = [
  "current_cgpa", "cgpa_trend", "attendance_pct", "backlogs",
  "study_hours_week", "projects_count", "internships_count",
  "coding_practice_hours", "communication_score", "extracurricular_score"
];

// Wire up sliders to their live value labels
FEATURES.forEach(f => {
  const input = document.getElementById(f);
  const label = document.getElementById("v_" + f);
  input.addEventListener("input", () => { label.textContent = input.value; });
});

const analyzeBtn = document.getElementById("analyzeBtn");
const btnLabel = analyzeBtn.querySelector(".btn-label");
const btnLoader = analyzeBtn.querySelector(".btn-loader");
const errorMsg = document.getElementById("errorMsg");
const resultPanel = document.getElementById("resultPanel");

const riskBadge = document.getElementById("riskBadge");
const riskLabel = document.getElementById("riskLabel");
const riskConfidence = document.getElementById("riskConfidence");
const probBars = document.getElementById("probBars");
const gaugeFill = document.getElementById("gaugeFill");
const readinessValue = document.getElementById("readinessValue");
const factorsList = document.getElementById("factorsList");

const GAUGE_CIRCUMFERENCE = 327; // 2 * PI * 52, matches SVG r=52

analyzeBtn.addEventListener("click", async () => {
  errorMsg.hidden = true;
  const payload = {};
  FEATURES.forEach(f => { payload[f] = parseFloat(document.getElementById(f).value); });

  setLoading(true);
  resultPanel.hidden = true;
  riskBadge.classList.remove("reveal", "safe", "critical");

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    if (!res.ok) {
      errorMsg.textContent = data.error || "Something went wrong.";
      errorMsg.hidden = false;
      return;
    }

    renderResult(data);
  } catch (err) {
    errorMsg.textContent = "Could not reach the server. Is app.py running?";
    errorMsg.hidden = false;
  } finally {
    setLoading(false);
  }
});

function setLoading(isLoading) {
  analyzeBtn.disabled = isLoading;
  btnLoader.hidden = !isLoading;
  btnLabel.textContent = isLoading ? "Analyzing..." : "Analyze my standing";
}

function renderResult(data) {
  resultPanel.hidden = false;

  riskLabel.textContent = data.risk_level.toUpperCase();
  riskConfidence.textContent = `${data.confidence}% model confidence`;

  if (data.risk_level === "Safe") riskBadge.classList.add("safe");
  if (data.risk_level === "Critical") riskBadge.classList.add("critical");
  requestAnimationFrame(() => riskBadge.classList.add("reveal"));

  // Probability bars
  probBars.innerHTML = "";
  ["Safe", "Warning", "Critical"].forEach(cls => {
    const pct = data.probabilities[cls] ?? 0;
    const row = document.createElement("div");
    row.className = "prob-bar-row";
    row.innerHTML = `
      <span style="width:56px">${cls}</span>
      <span class="prob-bar-track"><span class="prob-bar-fill ${cls}" style="width:0%"></span></span>
      <span style="width:38px">${pct}%</span>
    `;
    probBars.appendChild(row);
    setTimeout(() => { row.querySelector(".prob-bar-fill").style.width = pct + "%"; }, 100);
  });

  // Gauge
  const readiness = data.placement_readiness;
  readinessValue.textContent = Math.round(readiness);
  const offset = GAUGE_CIRCUMFERENCE - (readiness / 100) * GAUGE_CIRCUMFERENCE;
  gaugeFill.style.strokeDashoffset = GAUGE_CIRCUMFERENCE;
  setTimeout(() => { gaugeFill.style.strokeDashoffset = offset; }, 150);

  // Top factors — wording depends on whether this is a risk or a safe profile
  document.querySelector(".factors-label").textContent =
    data.risk_level === "Safe" ? "Where you still have room to grow" : "Top factors driving this result";

  factorsList.innerHTML = "";
  data.top_factors.forEach(f => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="factor-name">${f.feature}</span><span class="factor-rec">${f.recommendation}</span>`;
    factorsList.appendChild(li);
  });

  resultPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
