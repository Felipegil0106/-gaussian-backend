"""
Backend API + Página Web para Gaussian Scanner
Orquesta: Página Web → R2 → RunPod Serverless → R2

NUEVO en esta versión:
  - Sirve una página web en "/" para subir ZIPs (drag & drop)
  - Guarda el log completo del worker cuando hay error
  - Endpoint GET /api/jobs/{id}/log para descargar el log

Endpoints:
  GET  /                       → Página web (testeador de renderizado)
  POST /api/jobs               → Crea job, sube ZIP a R2, dispara RunPod
  GET  /api/jobs/{id}          → Estado del job
  GET  /api/jobs/{id}/download → URLs de descarga del resultado
  GET  /api/jobs/{id}/log      → Descargar log del worker (texto)
  POST /api/webhooks/runpod    → RunPod notifica que terminó
  GET  /api/health             → Health check
"""

import os
import uuid
import json
import sqlite3
from datetime import datetime, timezone
from contextlib import contextmanager

import boto3
from botocore.config import Config
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
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

app = FastAPI(title="Gaussian Scanner API", version="2.0.0")

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
                glb_key TEXT,
                worker_log TEXT
            )
        """)
        # Migración: agregar worker_log si la tabla ya existía sin esa columna
        try:
            db.execute("ALTER TABLE jobs ADD COLUMN worker_log TEXT")
        except sqlite3.OperationalError:
            pass  # ya existe

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

# ── Cliente R2 ─────────────────────────────────────────────────

def get_r2():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

def r2_presigned_put(key, expires=7200):
    return get_r2().generate_presigned_url(
        "put_object", Params={"Bucket": R2_BUCKET, "Key": key}, ExpiresIn=expires)

def r2_presigned_get(key, expires=7200):
    return get_r2().generate_presigned_url(
        "get_object", Params={"Bucket": R2_BUCKET, "Key": key}, ExpiresIn=expires)

def r2_upload_file(file_obj, key):
    get_r2().upload_fileobj(file_obj, R2_BUCKET, key)

# ── Cliente RunPod ─────────────────────────────────────────────

RUNPOD_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"
RUNPOD_HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json",
}

def trigger_runpod(job_id, quality, zip_key, ply_key, glb_key):
    payload = {
        "input": {
            "download_url": r2_presigned_get(zip_key),
            "upload_url_ply": r2_presigned_put(ply_key),
            "upload_url_glb": r2_presigned_put(glb_key),
            "webhook_url": f"{BACKEND_URL}/api/webhooks/runpod",
            "job_id": job_id,
            "quality": quality,
        }
    }
    resp = requests.post(f"{RUNPOD_URL}/run", headers=RUNPOD_HEADERS, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()

def poll_runpod(runpod_job_id):
    resp = requests.get(f"{RUNPOD_URL}/status/{runpod_job_id}",
                       headers=RUNPOD_HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()

# ── Eventos ────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    init_db()
    print(f"Backend v2 iniciado. RunPod: {RUNPOD_ENDPOINT_ID}, R2: {R2_BUCKET}")

# ══════════════════════════════════════════════════════════════
# PÁGINA WEB (la interfaz para subir ZIPs)
# ══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE

# ══════════════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}

@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), quality: str = Form("fast")):
    if quality not in ("fast", "balanced", "quality"):
        raise HTTPException(400, "quality debe ser: fast, balanced, quality")

    job_id = str(uuid.uuid4())[:12]
    now = datetime.now(timezone.utc).isoformat()
    zip_key = f"uploads/{job_id}/input.zip"
    ply_key = f"results/{job_id}/scene.ply"
    glb_key = f"results/{job_id}/collision.glb"

    with get_db() as db:
        db.execute(
            "INSERT INTO jobs (id,status,quality,created_at,updated_at,ply_key,glb_key) "
            "VALUES (?,?,?,?,?,?,?)",
            (job_id, "uploading", quality, now, now, ply_key, glb_key))

    # Subir ZIP a R2
    try:
        r2_upload_file(file.file, zip_key)
    except Exception as e:
        with get_db() as db:
            db.execute("UPDATE jobs SET status='error', error=? WHERE id=?",
                      (f"Upload a R2 falló: {e}", job_id))
        raise HTTPException(500, f"Error subiendo a R2: {e}")

    # Disparar RunPod
    try:
        rp = trigger_runpod(job_id, quality, zip_key, ply_key, glb_key)
        runpod_job_id = rp.get("id", "")
    except Exception as e:
        with get_db() as db:
            db.execute("UPDATE jobs SET status='error', error=? WHERE id=?",
                      (f"RunPod trigger falló: {e}", job_id))
        raise HTTPException(500, f"Error disparando RunPod: {e}")

    with get_db() as db:
        db.execute("UPDATE jobs SET status='processing', runpod_job_id=?, updated_at=? WHERE id=?",
                  (runpod_job_id, datetime.now(timezone.utc).isoformat(), job_id))

    return {"job_id": job_id, "status": "processing", "quality": quality}

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with get_db() as db:
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Job no encontrado")
        job = dict(row)

    # Si está procesando, consultar RunPod
    if job["status"] == "processing" and job.get("runpod_job_id"):
        try:
            rp = poll_runpod(job["runpod_job_id"])
            rp_status = rp.get("status", "")

            if rp_status == "COMPLETED":
                output = rp.get("output", {})
                if isinstance(output, dict) and output.get("status") == "success":
                    with get_db() as db:
                        db.execute(
                            "UPDATE jobs SET status='completed', frames_used=?, ply_mb=?, "
                            "has_collision=?, seconds=?, updated_at=? WHERE id=?",
                            (output.get("frames_used",0), output.get("ply_mb",0),
                             1 if output.get("has_collision") else 0, output.get("seconds",0),
                             datetime.now(timezone.utc).isoformat(), job_id))
                    job.update(status="completed", frames_used=output.get("frames_used",0),
                              ply_mb=output.get("ply_mb",0), seconds=output.get("seconds",0))
                else:
                    # ERROR: guardar log completo del worker
                    err_msg = output.get("error","Error desconocido") if isinstance(output,dict) else str(output)
                    worker_log = build_worker_log(output) if isinstance(output, dict) else str(output)
                    with get_db() as db:
                        db.execute("UPDATE jobs SET status='error', error=?, worker_log=?, updated_at=? WHERE id=?",
                                  (err_msg, worker_log, datetime.now(timezone.utc).isoformat(), job_id))
                    job.update(status="error", error=err_msg, worker_log=worker_log)

            elif rp_status == "FAILED":
                output = rp.get("output", {})
                worker_log = build_worker_log(output) if isinstance(output, dict) else "RunPod job failed"
                with get_db() as db:
                    db.execute("UPDATE jobs SET status='error', error=?, worker_log=?, updated_at=? WHERE id=?",
                              ("RunPod job failed", worker_log, datetime.now(timezone.utc).isoformat(), job_id))
                job.update(status="error", worker_log=worker_log)
        except Exception:
            pass

    return {
        "job_id": job["id"],
        "status": job["status"],
        "quality": job.get("quality"),
        "frames_used": job.get("frames_used"),
        "ply_mb": job.get("ply_mb"),
        "has_collision": bool(job.get("has_collision")),
        "seconds": job.get("seconds"),
        "error": job.get("error"),
        "has_log": bool(job.get("worker_log")),
    }

@app.get("/api/jobs/{job_id}/download")
def download_result(job_id: str):
    with get_db() as db:
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Job no encontrado")
        job = dict(row)
    if job["status"] != "completed":
        raise HTTPException(400, f"Job no está listo. Estado: {job['status']}")
    result = {"job_id": job_id, "ply_url": r2_presigned_get(job["ply_key"]), "ply_mb": job.get("ply_mb",0)}
    if job.get("has_collision"):
        result["glb_url"] = r2_presigned_get(job["glb_key"])
    return result

@app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
def get_log(job_id: str):
    """Devuelve el log del worker como texto plano (para descargar)."""
    with get_db() as db:
        row = db.execute("SELECT worker_log, error, status, quality, frames_used FROM jobs WHERE id=?",
                        (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Job no encontrado")
    log = row["worker_log"] or ""
    header = (
        f"========================================\n"
        f"LOG DE RENDERIZADO - Job: {job_id}\n"
        f"Estado: {row['status']}\n"
        f"Calidad: {row['quality']}\n"
        f"Error: {row['error'] or 'N/A'}\n"
        f"========================================\n\n"
    )
    return header + (log if log else "(Sin log detallado disponible)")

@app.post("/api/webhooks/runpod")
async def runpod_webhook(request: Request):
    try:
        data = await request.json()
    except:
        return {"ok": False}

    job_id = data.get("job_id", "")
    status = data.get("status", "")

    with get_db() as db:
        if status == "success":
            db.execute(
                "UPDATE jobs SET status='completed', frames_used=?, ply_mb=?, "
                "has_collision=?, seconds=?, updated_at=? WHERE id=?",
                (data.get("frames_used",0), data.get("ply_mb",0),
                 1 if data.get("has_collision") else 0, data.get("seconds",0),
                 datetime.now(timezone.utc).isoformat(), job_id))
        else:
            worker_log = build_worker_log(data)
            db.execute("UPDATE jobs SET status='error', error=?, worker_log=?, updated_at=? WHERE id=?",
                      (data.get("error","Error desconocido"), worker_log,
                       datetime.now(timezone.utc).isoformat(), job_id))
    return {"ok": True}

# ── Utilidad para construir el log legible ─────────────────────

def build_worker_log(output: dict) -> str:
    """Construye un log legible a partir del output del worker."""
    parts = []
    if output.get("stage"):
        parts.append(f"ETAPA DONDE FALLÓ: {output['stage']}")
    if output.get("error"):
        parts.append(f"\nMENSAJE DE ERROR:\n{output['error']}")
    if output.get("last_cmd"):
        parts.append(f"\nÚLTIMO COMANDO EJECUTADO:\n{output['last_cmd']}")
    if output.get("log"):
        parts.append(f"\nLOG DEL PIPELINE:\n{output['log']}")
    if output.get("traceback"):
        parts.append(f"\nTRACEBACK COMPLETO:\n{output['traceback']}")
    return "\n".join(parts) if parts else json.dumps(output, indent=2, default=str)

# ══════════════════════════════════════════════════════════════
# HTML DE LA PÁGINA
# ══════════════════════════════════════════════════════════════

HTML_PAGE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Testeador de Renderizado 3D</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%);
    color: #eee; min-height: 100vh; padding: 20px;
  }
  .container { max-width: 700px; margin: 0 auto; }
  h1 { text-align:center; font-size: 28px; margin-bottom: 8px;
       background: linear-gradient(90deg,#FF6B35,#00D9FF);
       -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  .subtitle { text-align:center; color:#888; margin-bottom: 30px; font-size:14px; }
  .card { background:#1a1a1a; border-radius:16px; padding:24px; margin-bottom:20px;
          border:1px solid #2a2a2a; }
  #dropzone {
    border: 2px dashed #FF6B35; border-radius:12px; padding:48px 24px;
    text-align:center; cursor:pointer; transition:.2s; background:#1f1f1f;
  }
  #dropzone:hover, #dropzone.drag { background:#2a2018; border-color:#00D9FF; }
  #dropzone .icon { font-size:48px; margin-bottom:12px; }
  #dropzone .text { font-size:16px; color:#ccc; }
  #dropzone .sub { font-size:13px; color:#777; margin-top:8px; }
  .file-info { background:#0f2a0f; border:1px solid #2a5a2a; border-radius:8px;
               padding:12px; margin-top:16px; display:none; }
  select, button {
    width:100%; padding:14px; border-radius:10px; border:none; font-size:15px;
    margin-top:12px; cursor:pointer;
  }
  select { background:#2a2a2a; color:#eee; }
  .btn-primary { background:linear-gradient(90deg,#FF6B35,#C84B1A); color:#fff;
                 font-weight:bold; font-size:16px; }
  .btn-primary:disabled { opacity:.4; cursor:not-allowed; }
  .btn-download { background:#00D9FF; color:#000; font-weight:bold; }
  .btn-success { background:#4CAF50; color:#fff; font-weight:bold; }
  #progress { display:none; }
  .log-box { background:#0a0a0a; border:1px solid #2a2a2a; border-radius:8px;
             padding:16px; font-family:monospace; font-size:12px; color:#9fef9f;
             height:240px; overflow-y:auto; white-space:pre-wrap; margin-top:16px; }
  .status { text-align:center; padding:16px; font-size:16px; font-weight:bold; }
  .status.processing { color:#00D9FF; }
  .status.success { color:#4CAF50; }
  .status.error { color:#E53935; }
  .spinner { display:inline-block; width:16px; height:16px; border:3px solid #333;
             border-top-color:#00D9FF; border-radius:50%; animation:spin 1s linear infinite;
             vertical-align:middle; margin-right:8px; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .hidden { display:none !important; }
</style>
</head>
<body>
<div class="container">
  <h1>🎨 Testeador de Renderizado 3D</h1>
  <p class="subtitle">Sube tu ZIP de fotos → Renderiza → Descarga el log si hay error</p>

  <div class="card" id="upload-card">
    <div id="dropzone">
      <div class="icon">📦</div>
      <div class="text">Arrastra tu ZIP aquí o haz click</div>
      <div class="sub">Archivo .zip con fotos (mínimo 20)</div>
    </div>
    <input type="file" id="fileInput" accept=".zip" style="display:none">
    <div class="file-info" id="fileInfo"></div>

    <select id="quality">
      <option value="fast">Rápido (~10 min) - para probar</option>
      <option value="balanced">Balanceado (~25 min)</option>
      <option value="quality">Máxima calidad (~45 min)</option>
    </select>

    <button class="btn-primary" id="renderBtn" disabled>🚀 Iniciar Renderizado</button>
  </div>

  <div class="card" id="progress">
    <div class="status processing" id="statusText">
      <span class="spinner"></span>Procesando...
    </div>
    <div class="log-box" id="logBox">Iniciando...</div>
    <div id="resultActions" class="hidden">
      <button class="btn-success hidden" id="viewBtn">🎨 Ver render en superspl.at</button>
      <button class="btn-download hidden" id="logBtn">📄 Descargar log del error</button>
      <button class="btn-primary" id="newBtn">🔄 Probar otro ZIP</button>
    </div>
  </div>
</div>

<script>
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const renderBtn = document.getElementById('renderBtn');
const quality = document.getElementById('quality');
const uploadCard = document.getElementById('upload-card');
const progress = document.getElementById('progress');
const statusText = document.getElementById('statusText');
const logBox = document.getElementById('logBox');
const resultActions = document.getElementById('resultActions');
const viewBtn = document.getElementById('viewBtn');
const logBtn = document.getElementById('logBtn');
const newBtn = document.getElementById('newBtn');

let selectedFile = null;
let currentJobId = null;
let pollTimer = null;

// Drag & drop
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
dropzone.addEventListener('drop', e => {
  e.preventDefault(); dropzone.classList.remove('drag');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', e => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
  if (!file.name.toLowerCase().endsWith('.zip')) {
    alert('Por favor sube un archivo .zip');
    return;
  }
  selectedFile = file;
  const mb = (file.size / 1024 / 1024).toFixed(1);
  fileInfo.style.display = 'block';
  fileInfo.textContent = `✓ ${file.name} (${mb} MB)`;
  renderBtn.disabled = false;
}

function addLog(msg) {
  logBox.textContent += '\\n' + msg;
  logBox.scrollTop = logBox.scrollHeight;
}

renderBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  uploadCard.classList.add('hidden');
  progress.style.display = 'block';
  resultActions.classList.add('hidden');
  logBox.textContent = 'Subiendo ZIP al servidor...';
  statusText.className = 'status processing';
  statusText.innerHTML = '<span class="spinner"></span>Subiendo...';

  const formData = new FormData();
  formData.append('file', selectedFile);
  formData.append('quality', quality.value);

  try {
    const resp = await fetch('/api/jobs', { method:'POST', body: formData });
    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error('Error ' + resp.status + ': ' + txt);
    }
    const data = await resp.json();
    currentJobId = data.job_id;
    addLog('✓ ZIP subido. Job ID: ' + currentJobId);
    addLog('✓ Renderizado iniciado en RunPod (RTX 4090)');
    addLog('Esperando resultado... (15-45 min según calidad)');
    statusText.innerHTML = '<span class="spinner"></span>Procesando en GPU...';
    startPolling();
  } catch (err) {
    addLog('❌ ERROR: ' + err.message);
    statusText.className = 'status error';
    statusText.textContent = '❌ Error al subir';
    showNewButton();
  }
});

function startPolling() {
  let elapsed = 0;
  pollTimer = setInterval(async () => {
    elapsed += 10;
    try {
      const resp = await fetch('/api/jobs/' + currentJobId);
      const job = await resp.json();
      statusText.innerHTML = '<span class="spinner"></span>Procesando... ' + elapsed + 's';

      if (job.status === 'completed') {
        clearInterval(pollTimer);
        addLog('');
        addLog('✅ RENDERIZADO COMPLETADO');
        addLog('Frames usados: ' + (job.frames_used || '?'));
        addLog('Tamaño: ' + (job.ply_mb || '?') + ' MB');
        addLog('Tiempo: ' + (job.seconds || '?') + 's');
        statusText.className = 'status success';
        statusText.textContent = '✅ ¡Renderizado completado!';
        showSuccess();
      } else if (job.status === 'error') {
        clearInterval(pollTimer);
        addLog('');
        addLog('❌ RENDERIZADO FALLÓ');
        addLog('Error: ' + (job.error || 'desconocido'));
        statusText.className = 'status error';
        statusText.textContent = '❌ Falló el renderizado';
        showError(job.has_log);
      }
    } catch (err) {
      addLog('⚠ Error consultando estado: ' + err.message);
    }
  }, 10000);
}

function showSuccess() {
  resultActions.classList.remove('hidden');
  viewBtn.classList.remove('hidden');
  logBtn.classList.add('hidden');
  viewBtn.onclick = async () => {
    const resp = await fetch('/api/jobs/' + currentJobId + '/download');
    const data = await resp.json();
    // Abrir superspl.at — el usuario tendrá que subir el archivo manualmente
    addLog('URL del PLY: ' + data.ply_url);
    window.open('https://superspl.at/editor', '_blank');
    addLog('Descarga el PLY desde el link de arriba y súbelo a superspl.at');
  };
  showNewButton();
}

function showError(hasLog) {
  resultActions.classList.remove('hidden');
  viewBtn.classList.add('hidden');
  if (hasLog) {
    logBtn.classList.remove('hidden');
    logBtn.onclick = () => {
      window.open('/api/jobs/' + currentJobId + '/log', '_blank');
    };
  }
  showNewButton();
}

function showNewButton() {
  resultActions.classList.remove('hidden');
  newBtn.onclick = () => location.reload();
}
</script>
</body>
</html>"""

# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
