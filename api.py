from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from shared import get_progress, update_progress
from training_main import run_federated_training

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status")
def read_status():
    return get_progress()

@app.post("/train")
def start_training(background_tasks: BackgroundTasks, defense: str = Query("none")):
    update_progress("status", "running")
    update_progress("current_round", 0)
    update_progress("accuracy", [])
    update_progress("mia_auc", [])
    update_progress("message", f"Training started with defense: {defense}")
    background_tasks.add_task(run_federated_training, defense)
    return {"detail": f"Training started with defense: {defense}"}

@app.get("/plot/miacurve")
def get_miacurve():
    filename = "static/miacurve.png" if not get_progress().get("defense") else f"static/miacurve_{get_progress()['defense']}.png"
    return FileResponse(filename, media_type="image/png")

@app.get("/plot/miaroc")
def get_miaroc():
    filename = "static/miaroc.png" if not get_progress().get("defense") else f"static/miaroc_{get_progress()['defense']}.png"
    return FileResponse(filename, media_type="image/png")
