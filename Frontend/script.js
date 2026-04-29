const fileInput = document.getElementById('fileInput');
const predictBtn = document.getElementById('predictBtn');
const resultsGrid = document.getElementById('resultsGrid');

let selectedFiles = [];

fileInput.addEventListener('change', (event) => {
    selectedFiles = Array.from(event.target.files);
    if (selectedFiles.length > 0) {
        predictBtn.disabled = false;
        resultsGrid.style.display = 'none';
        resultsGrid.innerHTML = ''; // Clear old results
    }
});

predictBtn.addEventListener('click', async () => {
    if (selectedFiles.length === 0) return;

    predictBtn.textContent = 'Processing Batch...';
    predictBtn.disabled = true;

    const formData = new FormData();
    // Append every file to the 'files' key so FastAPI reads it as a list
    selectedFiles.forEach(file => {
        formData.append('files', file); 
    });

    try {
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) throw new Error("Backend connection failed.");

        const data = await response.json();
        resultsGrid.innerHTML = ''; 
        resultsGrid.style.display = 'grid';

        // Loop through the batch results and create a card for each
        data.results.forEach((item, index) => {
            const fileUrl = URL.createObjectURL(selectedFiles[index]);
            
            let tagsHtml = item.predictions.length > 0 
                ? item.predictions.map(tag => `<span class="tag">${tag}</span>`).join('')
                : '<span>No objects detected.</span>';

            const card = `
                <div class="result-card">
                    <img src="${fileUrl}" alt="Preview" class="image-preview" />
                    <p class="filename">${item.filename}</p>
                    <div class="tags">${tagsHtml}</div>
                </div>
            `;
            resultsGrid.innerHTML += card;
        });

    } catch (error) {
        console.error(error);
        alert("Failed to connect. Is the FastAPI backend running?");
    } finally {
        predictBtn.textContent = 'Predict All';
        predictBtn.disabled = false;
    }
});