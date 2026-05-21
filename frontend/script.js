// ── Configuration ──
// For local development, change this to 'http://127.0.0.1:5000'
// For production (Render), set to your deployed backend URL
const BACKEND_URL = 'https://dishventory-ai-backend.onrender.com';

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('upload-form');
  const forecastResultsDiv = document.getElementById('forecast-results');
  const rawMaterialsDiv = document.getElementById('raw-materials-suggested');
  const submitBtn = document.getElementById('submit-btn');

  form.addEventListener('submit', async (event) => {
    event.preventDefault();

    const fileInput = document.getElementById('data-file');
    if (fileInput.files.length === 0) {
      alert('Please select a file to upload.');
      return;
    }

    // Show loading state
    submitBtn.disabled = true;
    submitBtn.textContent = '⏳ Processing...';
    forecastResultsDiv.innerHTML = '<p>Loading forecast... This may take a moment.</p>';
    rawMaterialsDiv.textContent = 'Calculating raw materials...';

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
        textDiv.textContent = `PEPPERONI M: ${result.predicted_sales} units (next 7 days)`;
        forecastResultsDiv.appendChild(textDiv);
      }

      if (result.graph_base64) {
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${result.graph_base64}`;
        img.alt = '7-Day Forecast Chart';
        forecastResultsDiv.appendChild(img);
      }

      if (result.ingredients_needed) {
        let output = '';
        for (const [key, value] of Object.entries(result.ingredients_needed)) {
          const label = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
          output += `${label}: ${value}\n`;
        }
        rawMaterialsDiv.textContent = output;
      }
    } catch (error) {
      forecastResultsDiv.innerHTML = `<p style="color: #e74c3c;">❌ Error: ${error.message}</p>`;
      rawMaterialsDiv.textContent = 'Failed to load raw materials.';
    } finally {
      // Reset button
      submitBtn.disabled = false;
      submitBtn.textContent = 'Upload and Predict';
    }
  });
});
