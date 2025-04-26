from fastapi import APIRouter, Depends, HTTPException, Header
import os, json

router = APIRouter(prefix="/config", tags=["config"])

def get_user_workspace(x_username: str = Header(...)):
    ws = f"user_data/{x_username}"
    os.makedirs(ws, exist_ok=True)
    return ws

@router.get("/")
def read_config(ws=Depends(get_user_workspace)):
    cfg_file = os.path.join(ws, "config.json")
    if not os.path.isfile(cfg_file): raise HTTPException(404, "Chưa có config")
    return json.load(open(cfg_file))

@router.post("/")
def write_config(config: dict, ws=Depends(get_user_workspace)):
    with open(os.path.join(ws, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    return {"status":"saved"}
