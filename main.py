"""
Backend API para Gaussian Scanner
Orquesta: App Android → R2 → RunPod → R2 → App Android

Endpoints:
  POST /api/jobs              → Crea job, sube ZIP a R2, dispara RunPod
  GET  /api/jobs/{id}         → Estado del job (polls RunPod)
  GET  /api/jobs/{id}/download → URLs para descargar .ply y .glb
  POST /api/webhooks/runpod   → RunPod notifica que terminó
  GET  /api/health            → Health check
"""

import os
import uuid
import time
import json
import sqlite3
import hashlib
import hmac
from datetime import datetime, timezone
from contextlib import contextmanager

import boto3
from botocore.config import Config
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import uvicorn

# ── Configuración desde variables de entorno ───────────────────

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "")

R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.environ.get("R2_SECRET_KEY", "")
R2_BUCKET = os.environ.get("R2_BUCKET", "gaussian-scanner")

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

PORT = int(os.environ.get("PORT", "8000"))

# ── App FastAPI ────────────────────────────────────────────────

app = FastAPI(title="Gaussian Scanner API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Base de datos SQLite ───────────────────────────────────────

DB_PATH = os.environ.get("DB_PATH", "/data/jobs.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT DEFAULT 'uploading',
                quality TEXT DEFAULT 'balanced',
                runpod_job_id TEXT,
                created_at TEXT,
                updated_at TEXT,
                frames_used INTEGER,
                ply_mb REAL,
                has_collision INTEGER DEFAULT 0,
                error TEXT,
                seconds REAL,
                ply_key TEXT,
                glb_key TEXT
            )
        """)

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

# ── Cliente R2 (S3-compatible) ─────────────────────────────────

def get_r2():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

def r2_presigned_put(key, expires=3600):
    """Genera URL para SUBIR archivo a R2 (PUT)."""
    return get_r2().generate_presigned_url(
        "put_object", Params={"Bucket": R2_BUCKET, "Key": key}, ExpiresIn=expires
    )

def r2_presigned_get(key, expires=3600):
    """Genera URL para DESCARGAR archivo de R2 (GET)."""
    return get_r2().generate_presigned_url(
        "get_object", Params={"Bucket": R2_BUCKET, "Key": key}, ExpiresIn=expires
    )

def r2_upload_file(file_obj, key):
    """Sube archivo directamente a R2."""
    get_r2().upload_fileobj(file_obj, R2_BUCKET, key)

# ── Cliente RunPod ─────────────────────────────────────────────

RUNPOD_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"
RUNPOD_HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json",
}

def trigger_runpod(job_id: str, quality: str, zip_key: str, ply_key: str, glb_key: str):
    """Lanza job en RunPod con URLs presigned para download/upload."""
    payload = {
        "input": {
            "download_url": r2_presigned_get(zip_key, expires=7200),
            "upload_url_ply": r2_presigned_put(ply_key, expires=7200),
            "upload_url_glb": r2_presigned_put(glb_key, expires=7200),
            "webhook_url": f"{BACKEND_URL}/api/webhooks/runpod",
            "job_id": job_id,
            "quality": quality,
        }
    }
    resp = requests.post(f"{RUNPOD_URL}/run", headers=RUNPOD_HEADERS, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()

def poll_runpod(runpod_job_id: str):
    """Consulta estado de un job en RunPod."""
    resp = requests.get(f"{RUNPOD_URL}/status/{runpod_job_id}",
                       headers=RUNPOD_HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()

def cancel_runpod(runpod_job_id: str):
    """Cancela un job en RunPod."""
    try:
        requests.post(f"{RUNPOD_URL}/cancel/{runpod_job_id}",
                     headers=RUNPOD_HEADERS, timeout=10)
    except: pass

# ── Endpoints ──────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    init_db()
    print(f"Backend iniciado en {BACKEND_URL}")
    print(f"RunPod endpoint: {RUNPOD_ENDPOINT_ID}")
    print(f"R2 bucket: {R2_BUCKET}")

@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}

@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    quality: str = Form("balanced"),
):
    """
    Recibe un archivo ZIP con fotos, lo sube a R2, y dispara RunPod.
    La app Android llama este endpoint.
    """
    if quality not in ("fast", "balanced", "quality"):
        raise HTTPException(400, "quality debe ser: fast, balanced, quality")

    job_id = str(uuid.uuid4())[:12]
    now = datetime.now(timezone.utc).isoformat()

    # Keys en R2
    zip_key = f"uploads/{job_id}/input.zip"
    ply_key = f"results/{job_id}/scene.ply"
    glb_key = f"results/{job_id}/collision.glb"

    # 1. Crear registro en DB
    with get_db() as db:
        db.execute(
            "INSERT INTO jobs (id,status,quality,created_at,updated_at,ply_key,glb_key) "
            "VALUES (?,?,?,?,?,?,?)",
            (job_id, "uploading", quality, now, now, ply_key, glb_key)
        )

    # 2. Subir ZIP a R2
    try:
        r2_upload_file(file.file, zip_key)
    except Exception as e:
        with get_db() as db:
            db.execute("UPDATE jobs SET status='error', error=? WHERE id=?",
                      (f"Upload a R2 falló: {e}", job_id))
        raise HTTPException(500, f"Error subiendo a R2: {e}")

    # 3. Disparar RunPod
    try:
        rp_resp = trigger_runpod(job_id, quality, zip_key, ply_key, glb_key)
        runpod_job_id = rp_resp.get("id", "")
    except Exception as e:
        with get_db() as db:
            db.execute("UPDATE jobs SET status='error', error=? WHERE id=?",
                      (f"RunPod trigger falló: {e}", job_id))
        raise HTTPException(500, f"Error disparando RunPod: {e}")

    # 4. Actualizar DB con RunPod job ID
    with get_db() as db:
        db.execute(
            "UPDATE jobs SET status='processing', runpod_job_id=?, updated_at=? WHERE id=?",
            (runpod_job_id, datetime.now(timezone.utc).isoformat(), job_id)
        )

    return {
        "job_id": job_id,
        "status": "processing",
        "quality": quality,
        "message": "Tu scan está procesándose. Consulta el estado con GET /api/jobs/{job_id}",
    }

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    """
    Devuelve estado del job. Si está en 'processing', consulta RunPod para actualizar.
    """
    with get_db() as db:
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Job no encontrado")
        job = dict(row)

    # Si está en processing, poll RunPod para ver si terminó
    if job["status"] == "processing" and job.get("runpod_job_id"):
        try:
            rp = poll_runpod(job["runpod_job_id"])
            rp_status = rp.get("status", "")

            if rp_status == "COMPLETED":
                output = rp.get("output", {})
                if isinstance(output, dict) and output.get("status") == "success":
                    with get_db() as db:
                        db.execute(
                            "UPDATE jobs SET status='completed', frames_used=?, "
                            "ply_mb=?, has_collision=?, seconds=?, updated_at=? WHERE id=?",
                            (output.get("frames_used",0), output.get("ply_mb",0),
                             1 if output.get("has_collision") else 0,
                             output.get("seconds",0),
                             datetime.now(timezone.utc).isoformat(), job_id)
                        )
                    job["status"] = "completed"
                    job["frames_used"] = output.get("frames_used",0)
                    job["ply_mb"] = output.get("ply_mb",0)
                    job["seconds"] = output.get("seconds",0)
                else:
                    error_msg = output.get("error","Error desconocido") if isinstance(output,dict) else str(output)
                    with get_db() as db:
                        db.execute("UPDATE jobs SET status='error', error=?, updated_at=? WHERE id=?",
                                  (error_msg, datetime.now(timezone.utc).isoformat(), job_id))
                    job["status"] = "error"
                    job["error"] = error_msg

            elif rp_status == "FAILED":
                with get_db() as db:
                    db.execute("UPDATE jobs SET status='error', error='RunPod job failed', updated_at=? WHERE id=?",
                              (datetime.now(timezone.utc).isoformat(), job_id))
                job["status"] = "error"

        except Exception as e:
            pass  # Si no podemos pollear, no es fatal, el webhook lo resolverá

    return {
        "job_id": job["id"],
        "status": job["status"],
        "quality": job.get("quality"),
        "created_at": job.get("created_at"),
        "frames_used": job.get("frames_used"),
        "ply_mb": job.get("ply_mb"),
        "has_collision": bool(job.get("has_collision")),
        "seconds": job.get("seconds"),
        "error": job.get("error"),
    }

@app.get("/api/jobs/{job_id}/download")
def download_result(job_id: str):
    """
    Devuelve URLs presigned para descargar los archivos resultado.
    Solo funciona si el job está en status 'completed'.
    """
    with get_db() as db:
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Job no encontrado")
        job = dict(row)

    if job["status"] != "completed":
        raise HTTPException(400, f"Job no está listo. Estado actual: {job['status']}")

    result = {
        "job_id": job_id,
        "ply_url": r2_presigned_get(job["ply_key"], expires=3600),
        "ply_mb": job.get("ply_mb", 0),
    }

    if job.get("has_collision"):
        result["glb_url"] = r2_presigned_get(job["glb_key"], expires=3600)

    return result

@app.post("/api/webhooks/runpod")
async def runpod_webhook(request: Request):
    """
    RunPod llama este endpoint cuando un job termina.
    Actualiza el estado del job en la DB.
    """
    try:
        data = await request.json()
    except:
        return {"ok": False}

    job_id = data.get("job_id", "")
    status = data.get("status", "")

    with get_db() as db:
        if status == "success":
            db.execute(
                "UPDATE jobs SET status='completed', frames_used=?, "
                "ply_mb=?, has_collision=?, seconds=?, updated_at=? WHERE id=?",
                (data.get("frames_used",0), data.get("ply_mb",0),
                 1 if data.get("has_collision") else 0,
                 data.get("seconds",0),
                 datetime.now(timezone.utc).isoformat(), job_id)
            )
        else:
            db.execute(
                "UPDATE jobs SET status='error', error=?, updated_at=? WHERE id=?",
                (data.get("error","Error desconocido"),
                 datetime.now(timezone.utc).isoformat(), job_id)
            )

    return {"ok": True}

@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """Cancela un job que está procesándose en RunPod."""
    with get_db() as db:
        row = db.execute("SELECT runpod_job_id FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Job no encontrado")
        if row["runpod_job_id"]:
            cancel_runpod(row["runpod_job_id"])
        db.execute("UPDATE jobs SET status='cancelled', updated_at=? WHERE id=?",
                  (datetime.now(timezone.utc).isoformat(), job_id))
    return {"status": "cancelled"}

# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
