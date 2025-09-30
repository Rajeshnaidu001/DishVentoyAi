document.addEventListener('DOMContentLoaded', () => {
  console.log('Script loaded!'); // for debugging

  const form = document.getElementById('upload-form');
  const forecastResultsDiv = document.getElementById('forecast-results');
  const rawMaterialsDiv = document.getElementById('raw-materials-suggested');

  form.addEventListener('submit', async (event) => {
    event.preventDefault(); // 🔴 This prevents the form from submitting normally

    forecastResultsDiv.innerHTML = '<p>Loading forecast...</p>';
    rawMaterialsDiv.textContent = 'Loading raw materials...';

    const fileInput = document.getElementById('data-file');
    if (fileInput.files.length === 0) {
      alert('Please select a file to upload.');
      forecastResultsDiv.innerHTML = '';
      rawMaterialsDiv.textContent = '';
      return;
    }

    const formData = new FormData();
    formData.append('data-file', fileInput.files[0]);

    try {
      const response = await fetch(
        'http://127.0.0.1:5000',
        {
          method: 'POST',
          body: formData,
          mode: 'no-cors',
          headers: {
            'Accept': 'application/json',
            'Content-Type': 'multipart/form-data'
          }
        }
      );

      const result = await response.json();
      forecastResultsDiv.innerHTML = '';

      if ('predicted_sales' in result) {
        forecastResultsDiv.textContent = `PEPPERONI M: ${result.predicted_sales} units`;
      }

      if (result.graph_base64) {
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${result.graph_base64}`;
        forecastResultsDiv.appendChild(img);
      }

      if (result.ingredients_needed) {
        let output = '';
        for (const [key, value] of Object.entries(result.ingredients_needed)) {
          output += `${key}: ${value}\n`;
        }
        rawMaterialsDiv.textContent = output;
      }
    } catch (error) {
      forecastResultsDiv.textContent = `Error: ${error.message}`;
      rawMaterialsDiv.textContent = 'Failed to load raw materials.';
    }
  });
});
