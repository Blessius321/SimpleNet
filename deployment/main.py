import model
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import os
import shutil
import uvicorn

app = FastAPI()

# make sure temps folder exists
os.makedirs("temps", exist_ok=True)

@app.post("/inference")
async def inference(file: UploadFile = File(...)):

    if file.filename is None:
        return JSONResponse(content={"defect": False, "message": "No file uploaded"})

    # Save file to temps/
    file_path = os.path.join("temps", f"input.{file.filename.split('.')[-1]}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Call model inference
    defect, result = model.inference(file_path)

    if defect and result:
        return FileResponse(result, media_type="image/jpeg")

    else:
        return JSONResponse(content={"defect": False, "message": "No defect detected"})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
