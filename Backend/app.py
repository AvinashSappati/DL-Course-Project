from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from inference import predict_batch

app = FastAPI(title="CNN-RNN Multi-Label Classifier")

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_multiple(files: List[UploadFile] = File(...)):
    try:
        # Read all files into a list of bytes
        image_bytes_list = [await file.read() for file in files]
        filenames = [file.filename for file in files]
        
        # Run batch prediction
        batch_results = predict_batch(image_bytes_list)
        
        # Zip filenames and predictions together
        final_output = [{"filename": f, "predictions": p} for f, p in zip(filenames, batch_results)]
        
        return {"results": final_output}
    except Exception as e:
        return {"error": str(e)}