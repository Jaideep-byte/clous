/*
 * dashboard.js
 * Fetches data from the Flask server and controls the dashboard.
 */

// We must wait for the HTML to be fully loaded before trying to find elements
document.addEventListener('DOMContentLoaded', () => {

  // --- (Chart setup, live data, historical data, and pending events...
  //     These functions remain UNCHANGED from the v3.0 AI version) ---
  // ... (pasting them here for completeness) ...
  const liveCharts = {};
  let historicalChart = null;
  const sensors = {
    tempChart: { label: "Temperature °C", key: "T", color: '#dc3545' },
    humChart: { label: "Humidity %", key: "H", color: '#007bff' },
    rainChart: { label: "Rain", key: "Rain", color: '#17a2b8' },
    soilChart: { label: "Soil", key: "Soil", color: '#8b4513' }
  };

  // Initialize all four live charts
  for (const [id, info] of Object.entries(sensors)) {
    const ctx = document.getElementById(id).getContext('2d');
    liveCharts[id] = new Chart(ctx, {
      type: "line",
      data: { labels: [], datasets: [{ label: info.label, data: [], fill: false, tension: 0.1, borderColor: info.color }] },
      options: { responsive: true, maintainAspectRatio: false, scales: { x: { display: false } } }
    });
  }

  async function refreshLiveData() {
    try {
      const res = await fetch('/data');
      const data = await res.json();
      if (!data.length) return;
      const latest = data[data.length - 1];
      const riskLabelEl = document.getElementById("risk-label");
      const confidenceEl = document.getElementById("confidence-label");
      const impactEl = document.getElementById("impact");
      const statusBox = document.getElementById("status");
      riskLabelEl.textContent = latest.Risk;
      confidenceEl.textContent = `${(latest.Confidence * 100).toFixed(0)}%`;
      let impactText = "--";
      
      // Updated logic for ImpactTime (assumes 'null' for pending, number for confirmed)
      if (latest.Risk.toLowerCase() !== "none") {
        if (latest.ImpactTime === 0.0 || latest.ImpactTime === 0) {
            // This now correctly checks the predicted time
            impactText = `Est. Impact: ~${latest.ImpactTime}h`;
        } else {
            // If it's a non-zero number, it's either predicted or confirmed
            impactText = `Est. Impact: ${latest.ImpactTime}h`;
        }
      }

      impactEl.textContent = impactText;
      riskLabelEl.className = "";

      // Apply styling based on risk
      if (latest.Risk.toLowerCase() === "none") {
        statusBox.style.background = "#e6ffe6"; // Greenish
        riskLabelEl.classList.add("risk-none");
      } else {
        statusBox.style.background = "#fff0b3"; // Yellowish
        riskLabelEl.classList.add("risk-pending");
      }
      
      // --- THIS IS THE FIX ---
      // Was: d.Timestamp (uppercase T)
      // Is:  d.timestamp (lowercase t)
      const labels = data.map(d => d.timestamp); 
      // --- END FIX ---

      for (const [id, info] of Object.entries(sensors)) {
        liveCharts[id].data.labels = labels;
        liveCharts[id].data.datasets[0].data = data.map(d => d[info.key]);
        liveCharts[id].update();
      }
    } catch (e) { console.error("Error refreshing live data:", e); }
  }

  async function loadHistoricalData(period = '24h', btnElement = null) {
    if (btnElement) {
      document.querySelectorAll('.hist-btn').forEach(btn => btn.classList.remove('active'));
      btnElement.classList.add('active');
    }
    try {
      const res = await fetch(`/historical_data?period=${period}`);
      const data = await res.json();
      if (historicalChart) { historicalChart.destroy(); }
      const ctx = document.getElementById('historicalChart').getContext('2d');
      historicalChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: data.labels,
          datasets: [
            { label: 'Avg Temp (°C)', data: data.avg_T, borderColor: '#dc3545', yAxisID: 'y' },
            { label: 'Avg Humidity (%)', data: data.avg_H, borderColor: '#007bff', yAxisID: 'y' },
            { label: 'Avg Soil', data: data.avg_Soil, borderColor: '#8b4513', yAxisID: 'y' },
            { label: 'Total Rain', data: data.total_Rain, borderColor: '#17a2b8', yAxisID: 'y1', type: 'bar' }
          ]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          scales: {
            x: { type: 'time', time: { unit: period === '24h' ? 'hour' : 'day' } },
            y: { type: 'linear', display: true, position: 'left', title: { display: true, text: 'Avg Values' } },
            y1: { type: 'linear', display: true, position: 'right', title: { display: true, text: 'Total Rain' }, grid: { drawOnChartArea: false } }
          }
        }
      });
    } catch (e) { console.error("Error loading historical data:", e); }
  }

  async function loadPendingEvents() {
    const tableBody = document.getElementById("pending-events-body");
    const loadingEl = document.getElementById("pending-loading");
    try {
      const res = await fetch('/pending_events');
      const events = await res.json();
      tableBody.innerHTML = "";
      loadingEl.textContent = events.length === 0 ? "No pending events found. Good job!" : "";
      events.forEach(event => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td>${event.timestamp}</td>
          <td>${event.Risk}</td>
          <td>${(event.Confidence * 100).toFixed(0)}%</td>
          <td>${event.T}</td>
          <td>${event.H}</td>
          <td>${event.Soil}</td>
          <td>${event.Rain}</td>
          <td><input type="number" step="0.1" min="0" class="impact-input" id="input-${event._id}"></td>
          <td><button class="save-btn" onclick="saveImpactTime('${event._id}')">Save</button></td>
        `;
        tableBody.appendChild(row);
      });
    } catch (e) { console.error("Error loading pending events:", e); }
  }

  async function saveImpactTime(docId) {
    const inputEl = document.getElementById(`input-${docId}`);
    const timeValue = inputEl.value;
    if (!timeValue || isNaN(parseFloat(timeValue)) || parseFloat(timeValue) < 0) {
      alert("Please enter a valid, positive number."); return;
    }
    try {
      const res = await fetch('/update_impact_time', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_id: docId, ground_truth_time: timeValue })
      });
      if (!res.ok) throw new Error(await res.text());
      alert("Success! Ground truth saved.");
      loadPendingEvents();
    } catch (e) { console.error("Error saving impact time:", e); alert("Error saving data."); }
  }

  // --- 4. NEW: Refresh Official Weather Alert ---
  async function refreshWeatherAlert() {
    try {
      const res = await fetch('/weather_alert');
      const alert = await res.json();
      
      // Format with line breaks and simple bolding
      document.getElementById('weather-alert-text').innerHTML = alert.text
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
      
      const timestamp = alert.timestamp ? new Date(alert.timestamp).toLocaleString() : '--';
      document.getElementById('weather-alert-timestamp').textContent = `Last checked: ${timestamp}`;
    } catch (e) {
      console.error("Error refreshing weather alert:", e);
      document.getElementById('weather-alert-text').textContent = "Error loading weather alerts.";
    }
  }

  /*
   * IMPORTANT:
   * We make 'loadHistoricalData' and 'saveImpactTime' global
   * so the 'onclick' attributes in the HTML can find them.
   */
  window.loadHistoricalData = loadHistoricalData;
  window.saveImpactTime = saveImpactTime;


  // --- 5. Initial Load and Refresh Timers ---
  refreshLiveData();
  refreshWeatherAlert(); 
  loadHistoricalData('24h', document.querySelector('.hist-btn'));
  loadPendingEvents();
  
  setInterval(refreshLiveData, 5000); // Live charts every 5 sec
  setInterval(refreshWeatherAlert, 60000); // Weather alert every 60 sec
  setInterval(loadPendingEvents, 60000); // Pending events every 60 sec

}); // End of DOMContentLoaded