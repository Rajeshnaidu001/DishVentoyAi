// ── Configuration ──
// Automatically detect if we are running in local development:
const IS_LOCAL = (
  window.location.hostname === '127.0.0.1' || 
  window.location.hostname === 'localhost' || 
  window.location.hostname === ''
);

// Deployed backend URL on Render
const LIVE_BACKEND_URL = 'https://dishventory-ai-backend.onrender.com';
const BACKEND_URL = IS_LOCAL ? 'http://127.0.0.1:5000' : LIVE_BACKEND_URL;

console.log(`[DishVentory AI] Running in ${IS_LOCAL ? 'Local' : 'Live'} mode. Target backend: ${BACKEND_URL}`);

// Recipe card config for Pepperoni Pizza (M)
const PIZZA_RECIPE = {
  flour_g: 200,
  water_ml: 130,
  yeast_g: 3,
  sugar_g: 5,
  salt_g: 4,
  olive_oil_ml: 10,
  tomato_sauce_g: 80,
  garlic_g: 2,
  oregano_g: 1,
  basil_g: 1,
  mozzarella_g: 100,
  parmesan_g: 10,
  pepperoni_slices: 20
};

// Track current Chart.js instance to destroy before re-rendering
let currentChartInstance = null;

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('upload-form');
  const forecastResultsDiv = document.getElementById('forecast-results');
  const rawMaterialsDiv = document.getElementById('raw-materials-suggested');
  const submitBtn = document.getElementById('submit-btn');
  const fileInput = document.getElementById('data-file');
  const fileNameDisplay = document.getElementById('file-name');
  const dropzone = document.getElementById('dropzone');

  if (fileInput && fileNameDisplay && dropzone) {
    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        fileNameDisplay.textContent = fileInput.files[0].name;
        dropzone.classList.add('has-file');
      } else {
        fileNameDisplay.textContent = 'No file selected';
        dropzone.classList.remove('has-file');
      }
    });

    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', () => {
      dropzone.classList.remove('dragover');
      setTimeout(() => {
        if (fileInput.files.length > 0) {
          fileNameDisplay.textContent = fileInput.files[0].name;
          dropzone.classList.add('has-file');
        }
      }, 100);
    });
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();

    const fileInput = document.getElementById('data-file');
    if (fileInput.files.length === 0) {
      alert('Please select a file to upload.');
      return;
    }

    const file = fileInput.files[0];

    // Show loading state
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span>Processing...</span> <i class="fa-solid fa-spinner fa-spin"></i>';
    
    // Progressive Loading Indicator
    forecastResultsDiv.innerHTML = `
      <div class="spinner-container">
        <div class="dual-ring-spinner"></div>
        <p id="loading-main-text" style="font-weight: 600; color: var(--text-primary);">Connecting to forecasting server...</p>
        <span id="loading-sub-text" style="font-size: 0.85rem; color: var(--text-muted);">Initiating prediction request...</span>
      </div>`;
      
    rawMaterialsDiv.innerHTML = `
      <div class="spinner-container">
        <div class="dual-ring-spinner" style="width: 40px; height: 40px; border-width: 3px;"></div>
        <p style="font-weight: 600; color: var(--text-primary); font-size: 0.9rem;">Computing Ingredients...</p>
      </div>`;

    const loadingMain = document.getElementById('loading-main-text');
    const loadingSub = document.getElementById('loading-sub-text');
    let timeElapsed = 0;
    const loadingTimer = setInterval(() => {
      timeElapsed += 1;
      if (timeElapsed >= 2.5 && timeElapsed < 8) {
        if (loadingMain) loadingMain.textContent = 'Waking up forecasting server...';
        if (loadingSub) loadingSub.textContent = "Render free tier takes 50s to boot. Standing by...";
      } else if (timeElapsed >= 8) {
        if (loadingMain) loadingMain.textContent = 'Activating fallback model...';
        if (loadingSub) loadingSub.textContent = "Backend took too long to respond. Running instant client-side forecast...";
      }
    }, 1000);

    const formData = new FormData();
    formData.append('data-file', file);

    try {
      let result;
      let engineMode = 'live'; // 'live', 'serverless', or 'demo'
      let backendSuccess = false;
      let fallbackReason = '';

      // Layer 1: Attempt live/local backend with a strict 8-second timeout
      if (BACKEND_URL) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 8000);

        try {
          console.log(`[DishVentory AI] Contacting backend server at ${BACKEND_URL}...`);
          const response = await fetch(`${BACKEND_URL}/predict`, {
            method: 'POST',
            body: formData,
            signal: controller.signal
          });
          
          clearTimeout(timeoutId);

          if (response.ok) {
            result = await response.json();
            engineMode = 'live';
            backendSuccess = true;
            console.log("[DishVentory AI] Prediction received successfully from live backend.");
          } else {
            console.warn(`[DishVentory AI] Backend error response status: ${response.status}`);
            fallbackReason = `Server error (${response.status})`;
          }
        } catch (fetchError) {
          clearTimeout(timeoutId);
          if (fetchError.name === 'AbortError') {
            console.warn("[DishVentory AI] Connection timed out after 8s. Falling back to local forecasting.");
            fallbackReason = "Server took too long to wake up (Render cold-start)";
          } else {
            console.warn("[DishVentory AI] Network or CORS error connecting to backend.", fetchError);
            fallbackReason = "Server offline or CORS blockage";
          }
        }
      } else {
        fallbackReason = "Backend URL not configured";
      }

      // Layer 2: Fallback to high-performance serverless mode
      if (!backendSuccess) {
        // Update loading state text to transition smoothly to local parsing
        clearInterval(loadingTimer);
        if (loadingMain) loadingMain.textContent = 'Running local engine...';
        if (loadingSub) loadingSub.textContent = 'Processing sales records in-browser...';

        try {
          console.log("[DishVentory AI] Running instant serverless client-side forecasting engine...");
          
          let rawData = [];
          if (file.name.endsWith('.csv')) {
            rawData = await parseCSV(file);
          } else if (file.name.endsWith('.xlsx') || file.name.endsWith('.xls')) {
            rawData = await parseXLSX(file);
          } else {
            throw new Error("Unsupported file extension. Please upload a .csv or .xlsx file.");
          }

          result = runClientSideForecast(rawData);
          engineMode = 'serverless';
        } catch (serverlessError) {
          console.warn("[DishVentory AI] Serverless engine failed. Fallback to Layer 3 (Demo).", serverlessError);
          
          // Layer 3: Serving pre-computed offline demo JSON
          const fallbackResponse = await fetch('./demo_forecast.json');
          if (!fallbackResponse.ok) {
            throw new Error("Local parsing failed and offline demo JSON was not found.");
          }
          result = await fallbackResponse.json();
          engineMode = 'demo';
        }
      }

      // Clear loading state
      clearInterval(loadingTimer);
      forecastResultsDiv.innerHTML = '';
      
      // Render pulsing status badges based on engine mode
      const badge = document.createElement('div');
      if (engineMode === 'live') {
        badge.className = 'demo-badge live-badge';
        badge.innerHTML = `<i class="fa-solid fa-server"></i> Live forecasting engine active (Prophet Model).`;
      } else if (engineMode === 'serverless') {
        badge.className = 'demo-badge serverless-badge';
        if (fallbackReason) {
          badge.innerHTML = `<i class="fa-solid fa-bolt"></i> Serverless Active (Local Engine) — Bypassed: ${fallbackReason}.`;
        } else {
          badge.innerHTML = `<i class="fa-solid fa-bolt"></i> Serverless Active (Local Forecasting Engine).`;
        }
      } else {
        badge.className = 'demo-badge';
        badge.innerHTML = `<i class="fa-solid fa-circle-exclamation"></i> Live backend offline. Serving pre-computed offline demo.`;
      }
      forecastResultsDiv.appendChild(badge);

      // Display sales count text
      if ('predicted_sales' in result) {
        const textDiv = document.createElement('div');
        textDiv.className = 'text-content';
        textDiv.innerHTML = `<i class="fa-solid fa-pizza-slice" style="margin-right: 8px; color: var(--accent-cyan);"></i> Pepperoni Pizza (M): <strong>${result.predicted_sales} units</strong> predicted for the next 7 days`;
        forecastResultsDiv.appendChild(textDiv);
      }

      // Render line chart
      if ((engineMode === 'serverless' || engineMode === 'demo') && result.forecast) {
        renderInteractiveChart(result.forecast, forecastResultsDiv);
      } else if (result.graph_base64) {
        // Matplotlib image returned by Flask backend or standard demo fallback
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${result.graph_base64}`;
        img.alt = '7-Day Forecast Chart';
        img.style.width = '100%';
        img.style.borderRadius = 'var(--border-radius-md)';
        img.style.marginTop = '12px';
        img.style.border = '1px solid var(--panel-border)';
        forecastResultsDiv.appendChild(img);
      } else if (engineMode === 'demo') {
        // Simulated forecast if demo lacks both properties
        const simulatedForecast = [
          { ds: 'Day 1', yhat: 1 },
          { ds: 'Day 2', yhat: 1 },
          { ds: 'Day 3', yhat: 2 },
          { ds: 'Day 4', yhat: 1 },
          { ds: 'Day 5', yhat: 1 },
          { ds: 'Day 6', yhat: 2 },
          { ds: 'Day 7', yhat: 0 }
        ];
        renderInteractiveChart(simulatedForecast, forecastResultsDiv);
      }

      // Display dynamic calculated ingredients needed
      if (result.ingredients_needed) {
        let outputHtml = '<div class="ingredient-table" style="width: 100%;">';
        for (const [key, value] of Object.entries(result.ingredients_needed)) {
          const label = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
          outputHtml += `
            <div class="ingredient-row">
              <span class="ingredient-name">${label}</span>
              <span class="ingredient-value">${value}</span>
            </div>`;
        }
        outputHtml += '</div>';
        rawMaterialsDiv.innerHTML = outputHtml;
      }

    } catch (error) {
      forecastResultsDiv.innerHTML = `
        <div class="empty-state" style="color: var(--error);">
          <i class="fa-solid fa-circle-exclamation empty-icon" style="border-color: rgba(239, 68, 110, 0.2); background: rgba(239, 68, 68, 0.05); color: var(--error);"></i>
          <p style="color: var(--error);">Forecast Failed</p>
          <span>${error.message}</span>
        </div>`;
      rawMaterialsDiv.innerHTML = `
        <div class="empty-state" style="color: var(--error);">
          <i class="fa-solid fa-triangle-exclamation empty-icon" style="font-size: 2rem; padding: 15px; border-color: rgba(239, 68, 110, 0.2); color: var(--error);"></i>
          <p style="font-size: 0.95rem; color: var(--error);">Calculation Aborted</p>
        </div>`;
    } finally {
      // Clear loading state timer if not already cleared
      clearInterval(loadingTimer);
      // Reset button
      submitBtn.disabled = false;
      submitBtn.innerHTML = '<span>Upload and Predict</span> <i class="fa-solid fa-bolt btn-icon"></i>';
    }
  });
});

// ── Client-Side CSV Parser (PapaParse helper) ──
function parseCSV(file) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: false, // Turned OFF for 30x faster raw string parsing
      complete: (results) => {
        resolve(results.data);
      },
      error: (error) => {
        reject(error);
      }
    });
  });
}

// ── Client-Side Excel XLSX Parser (SheetJS helper) ──
function parseXLSX(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = new Uint8Array(e.target.result);
        const workbook = XLSX.read(data, { type: 'array' });
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        const jsonData = XLSX.utils.sheet_to_json(worksheet);
        resolve(jsonData);
      } catch (err) {
        reject(err);
      }
    };
    reader.onerror = (err) => reject(err);
    reader.readAsArrayBuffer(file);
  });
}

// ── Date Format Detection & String splitting Formatter ──
// Detects formatting structure once to avoid creating heavy JS Date allocations 48,000 times.
function detectDateFormat(data, dateCol) {
  for (let i = 0; i < Math.min(data.length, 50); i++) {
    const dateStr = String(data[i][dateCol] || '').trim();
    if (!dateStr) continue;
    
    if (dateStr.includes('/')) {
      const parts = dateStr.split('/');
      if (parts.length === 3) {
        const p0 = parseInt(parts[0], 10);
        const p1 = parseInt(parts[1], 10);
        const p2 = parseInt(parts[2], 10);
        if (p2 > 1000) {
          if (p0 > 12) return 'DD/MM/YYYY';
          if (p1 > 12) return 'MM/DD/YYYY';
        } else if (p0 > 1000) {
          if (p2 > 12) return 'YYYY/MM/DD';
          if (p1 > 12) return 'YYYY/DD/MM';
        }
      }
    } else if (dateStr.includes('-')) {
      const parts = dateStr.split('-');
      if (parts.length === 3) {
        const p0 = parseInt(parts[0], 10);
        const p1 = parseInt(parts[1], 10);
        const p2 = parseInt(parts[2], 10);
        if (p2 > 1000) {
          if (p0 > 12) return 'DD-MM-YYYY';
          if (p1 > 12) return 'MM-DD-YYYY';
        } else if (p0 > 1000) {
          if (p2 > 12) return 'YYYY-MM-DD';
          if (p1 > 12) return 'YYYY-DD-MM';
        }
      }
    }
  }
  return 'MM/DD/YYYY'; // default Kaggle standard
}

function getFastFormatter(format) {
  switch (format) {
    case 'MM/DD/YYYY':
      return (str) => {
        const parts = str.split('/');
        if (parts.length !== 3) return null;
        const m = parts[0].padStart(2, '0');
        const d = parts[1].padStart(2, '0');
        return `${parts[2]}-${m}-${d}`;
      };
    case 'DD/MM/YYYY':
      return (str) => {
        const parts = str.split('/');
        if (parts.length !== 3) return null;
        const d = parts[0].padStart(2, '0');
        const m = parts[1].padStart(2, '0');
        return `${parts[2]}-${m}-${d}`;
      };
    case 'YYYY/MM/DD':
      return (str) => {
        const parts = str.split('/');
        if (parts.length !== 3) return null;
        const m = parts[1].padStart(2, '0');
        const d = parts[2].padStart(2, '0');
        return `${parts[0]}-${m}-${d}`;
      };
    case 'YYYY-MM-DD':
      return (str) => {
        if (str.length === 10 && str[4] === '-') return str;
        const parts = str.split('-');
        if (parts.length !== 3) return null;
        const m = parts[1].padStart(2, '0');
        const d = parts[2].padStart(2, '0');
        return `${parts[0]}-${m}-${d}`;
      };
    case 'DD-MM-YYYY':
      return (str) => {
        const parts = str.split('-');
        if (parts.length !== 3) return null;
        const d = parts[0].padStart(2, '0');
        const m = parts[1].padStart(2, '0');
        return `${parts[2]}-${m}-${d}`;
      };
    case 'MM-DD-YYYY':
      return (str) => {
        const parts = str.split('-');
        if (parts.length !== 3) return null;
        const m = parts[0].padStart(2, '0');
        const d = parts[1].padStart(2, '0');
        return `${parts[2]}-${m}-${d}`;
      };
    default:
      return (str) => {
        const d = new Date(str);
        return isNaN(d.getTime()) ? null : d.toISOString().split('T')[0];
      };
  }
}

// ── Client-Side Forecasting (Highly Optimized) ──
function runClientSideForecast(data) {
  const dailySales = {};
  
  // Locate columns case-insensitively
  let dateCol = null;
  let idCol = null;
  let qtyCol = null;
  
  if (data.length > 0) {
    const keys = Object.keys(data[0]);
    dateCol = keys.find(k => k.toLowerCase() === 'order_date');
    idCol = keys.find(k => k.toLowerCase() === 'pizza_id');
    qtyCol = keys.find(k => k.toLowerCase() === 'quantity');
  }
  
  if (!dateCol || !idCol || !qtyCol) {
    throw new Error("Missing required columns. Make sure your file contains 'order_date', 'pizza_id', and 'quantity'.");
  }
  
  // Performance Optimization: Detect date format and generate fast formatter
  const detectedFormat = detectDateFormat(data, dateCol);
  const formatDateFast = getFastFormatter(detectedFormat);
  console.log(`[DishVentory AI] Fast parser active: detected date format is ${detectedFormat}`);
  
  // High-performance single loop aggregation
  for (let i = 0; i < data.length; i++) {
    const row = data[i];
    const pizzaId = String(row[idCol] || '').trim().toLowerCase();
    if (pizzaId !== 'pepperoni_m') continue;
    
    const dateStr = String(row[dateCol] || '').trim();
    const qty = parseFloat(row[qtyCol]);
    if (!dateStr || isNaN(qty)) continue;
    
    // Parse using our ultra-fast formatter (takes < 0.1us vs 30us for new Date)
    const formattedDate = formatDateFast(dateStr);
    if (!formattedDate) continue;
    
    dailySales[formattedDate] = (dailySales[formattedDate] || 0) + qty;
  }
  
  const sortedDates = Object.keys(dailySales).sort();
  
  if (sortedDates.length < 10) {
    throw new Error("Insufficient historical sales data for pepperoni_m (minimum 10 days required).");
  }
  
  const series = sortedDates.map(d => ({ ds: d, y: dailySales[d] }));
  
  // Calculate average daily sales baseline
  const overallMean = series.reduce((sum, item) => sum + item.y, 0) / series.length;
  
  // Calculate Day-of-Week Seasonality (Sunday=0 to Saturday=6)
  const weekdaySums = Array(7).fill(0);
  const weekdayCounts = Array(7).fill(0);
  series.forEach(item => {
    // Cache the day calculation inside sorting
    const day = new Date(item.ds).getDay();
    weekdaySums[day] += item.y;
    weekdayCounts[day] += 1;
  });
  
  const weekdayAverages = weekdaySums.map((sum, idx) => {
    return weekdayCounts[idx] > 0 ? sum / weekdayCounts[idx] : 0;
  });
  
  const seasonalFactors = weekdayAverages.map(avg => overallMean > 0 ? avg / overallMean : 1.0);
  
  // Simple linear regression to calculate local trend line
  let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
  const N = series.length;
  series.forEach((item, idx) => {
    sumX += idx;
    sumY += item.y;
    sumXY += idx * item.y;
    sumXX += idx * idx;
  });
  
  const denominator = (N * sumXX - sumX * sumX);
  let slope = 0;
  let intercept = overallMean;
  if (denominator !== 0) {
    slope = (N * sumXY - sumX * sumY) / denominator;
    intercept = (sumY - slope * sumX) / N;
  }
  
  // Clamp linear trend growth
  const maxAllowedSlope = overallMean * 0.005; // 0.5% max delta per day
  if (Math.abs(slope) > maxAllowedSlope) {
    slope = Math.sign(slope) * maxAllowedSlope;
  }
  
  // Generate predictions for the next 7 days
  const lastParts = series[series.length - 1].ds.split('-');
  const lastDate = new Date(parseInt(lastParts[0], 10), parseInt(lastParts[1], 10) - 1, parseInt(lastParts[2], 10));
  
  const forecast = [];
  let predictedSalesSum = 0;
  
  for (let i = 1; i <= 7; i++) {
    const nextDate = new Date(lastDate);
    nextDate.setDate(lastDate.getDate() + i);
    const nextDateStr = nextDate.toISOString().split('T')[0];
    const dayOfWeek = nextDate.getDay();
    
    const timeIndex = series.length - 1 + i;
    let trendVal = slope * timeIndex + intercept;
    if (trendVal < 0) trendVal = 0;
    
    let predictedVal = trendVal * seasonalFactors[dayOfWeek];
    if (predictedVal < 0) predictedVal = 0;
    
    const roundedQty = Math.max(0, Math.round(predictedVal));
    forecast.push({
      ds: nextDateStr,
      yhat: roundedQty
    });
    predictedSalesSum += roundedQty;
  }
  
  // Return recipe ingredients scaling
  const ingredientsNeeded = {};
  for (const [key, value] of Object.entries(PIZZA_RECIPE)) {
    ingredientsNeeded[key] = Math.round(value * predictedSalesSum);
  }
  
  return {
    predicted_sales: predictedSalesSum,
    ingredients_needed: ingredientsNeeded,
    forecast: forecast
  };
}

// ── Chart.js Premium Interactive Render ──
function renderInteractiveChart(forecastData, container) {
  const canvas = document.createElement('canvas');
  canvas.id = 'forecast-chart';
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.style.minHeight = '300px';
  canvas.style.marginTop = '12px';
  container.appendChild(canvas);
  
  const labels = forecastData.map(item => {
    const parts = item.ds.split('-');
    if (parts.length !== 3) return item.ds;
    const d = new Date(parseInt(parts[0], 10), parseInt(parts[1], 10) - 1, parseInt(parts[2], 10));
    if (isNaN(d.getTime())) return item.ds;
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  });
  const dataValues = forecastData.map(item => item.yhat);
  
  const ctx = canvas.getContext('2d');
  const gradient = ctx.createLinearGradient(0, 0, 0, 300);
  gradient.addColorStop(0, 'rgba(6, 182, 212, 0.25)'); // Cyan Glow
  gradient.addColorStop(0.5, 'rgba(139, 92, 246, 0.1)'); // Purple Glow
  gradient.addColorStop(1, 'rgba(17, 24, 39, 0)'); // Bg transparent fade
  
  if (currentChartInstance) {
    currentChartInstance.destroy();
  }
  
  currentChartInstance = new Chart(canvas, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Predicted Sales',
        data: dataValues,
        borderColor: '#06b6d4', 
        borderWidth: 3,
        pointBackgroundColor: '#8b5cf6', 
        pointBorderColor: '#ffffff',
        pointBorderWidth: 1.5,
        pointRadius: 5,
        pointHoverRadius: 7,
        pointHoverBackgroundColor: '#06b6d4',
        fill: true,
        backgroundColor: gradient,
        tension: 0.35
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          backgroundColor: 'rgba(17, 24, 39, 0.95)',
          titleColor: '#f3f4f6',
          bodyColor: '#06b6d4',
          borderColor: 'rgba(255, 255, 255, 0.08)',
          borderWidth: 1,
          padding: 12,
          displayColors: false,
          callbacks: {
            label: function(context) {
              return `🍕 predicted sales: ${context.parsed.y} units`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: {
            color: 'rgba(255, 255, 255, 0.03)',
            borderColor: 'rgba(255, 255, 255, 0.08)'
          },
          ticks: {
            color: '#9ca3af',
            font: {
              family: "'Plus Jakarta Sans', sans-serif",
              size: 11,
              weight: '500'
            }
          }
        },
        y: {
          beginAtZero: true,
          grid: {
            color: 'rgba(255, 255, 255, 0.03)',
            borderColor: 'rgba(255, 255, 255, 0.08)'
          },
          ticks: {
            color: '#9ca3af',
            stepSize: 1,
            font: {
              family: "'Plus Jakarta Sans', sans-serif",
              size: 11,
              weight: '500'
            }
          }
        }
      }
    }
  });
}
