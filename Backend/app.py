from fastapi import FastAPI, File, UploadFile
from typing import List
from fastapi.staticfiles import StaticFiles
from inference import predict_batch

app = FastAPI(title="CNN-RNN API")

@app.post("/predict")
async def predict_multiple(files: List[UploadFile] = File(...)):
    try:
        image_bytes_list = [await file.read() for file in files]
        filenames = [file.filename for file in files]
        batch_results = predict_batch(image_bytes_list)
        final_output = [{"filename": f, "predictions": p} for f, p in zip(filenames, batch_results)]
        return {"results": final_output}
    except Exception as e:
        return {"error": str(e)}

# Pro-Move: Mount the static frontend folder to the root URL
app.mount("/", StaticFiles(directory="Frontend", html=True), name="frontend")
