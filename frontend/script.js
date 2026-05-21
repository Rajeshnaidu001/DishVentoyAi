// ── Configuration ──
// Automatically detect the environment and choose the correct backend URL:
// - Local development (localhost, 127.0.0.1, or local file): http://127.0.0.1:5000
// - Production (GitHub Pages): https://dishventory-ai-backend.onrender.com
const BACKEND_URL = (
  window.location.hostname === '127.0.0.1' || 
  window.location.hostname === 'localhost' || 
  window.location.hostname === ''
) 
  ? 'http://127.0.0.1:5000' 
  : 'https://dishventory-ai-backend.onrender.com';

console.log(`[DishVentory AI] Using backend at: ${BACKEND_URL}`);

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

    // Add dragover and dragleave visual effects
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

    // Show loading state
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span>Processing...</span> <i class="fa-solid fa-spinner fa-spin"></i>';
    
    forecastResultsDiv.innerHTML = `
      <div class="spinner-container">
        <div class="dual-ring-spinner"></div>
        <p style="font-weight: 600; color: var(--text-primary);">Running Prophet Forecasting Engine...</p>
        <span style="font-size: 0.85rem; color: var(--text-muted);">Analyzing historical sales trends & seasonal factors.</span>
      </div>`;
      
    rawMaterialsDiv.innerHTML = `
      <div class="spinner-container">
        <div class="dual-ring-spinner" style="width: 40px; height: 40px; border-width: 3px;"></div>
        <p style="font-weight: 600; color: var(--text-primary); font-size: 0.9rem;">Computing Ingredients...</p>
      </div>`;

    const formData = new FormData();
    formData.append('data-file', fileInput.files[0]);

    try {
      const response = await fetch(`${BACKEND_URL}/predict`, {
        method: 'POST',
        body: formData,
        // Do NOT set Content-Type — the browser sets it automatically with the correct boundary for FormData
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Server error: ${response.status}`);
      }

      const result = await response.json();
      forecastResultsDiv.innerHTML = '';

      if ('predicted_sales' in result) {
        const textDiv = document.createElement('div');
        textDiv.className = 'text-content';
        textDiv.innerHTML = `<i class="fa-solid fa-pizza-slice" style="margin-right: 8px; color: var(--accent-cyan);"></i> Pepperoni Pizza (M): <strong>${result.predicted_sales} units</strong> predicted for the next 7 days`;
        forecastResultsDiv.appendChild(textDiv);
      }

      if (result.graph_base64) {
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${result.graph_base64}`;
        img.alt = '7-Day Forecast Chart';
        forecastResultsDiv.appendChild(img);
      }

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
      // Reset button
      submitBtn.disabled = false;
      submitBtn.innerHTML = '<span>Upload and Predict</span> <i class="fa-solid fa-bolt btn-icon"></i>';
    }
  });
});
