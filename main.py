"""
Backend Gaussian Scanner v3 — Pod-Based GPU Orchestration
═══════════════════════════════════════════════════════════════════
Reemplaza la versión Serverless. Ahora alquila Pods on-demand:

  - Recibe ZIP de la app/web
  - Sube ZIP a R2
  - Alquila Pod RTX 4090 (cont 50GB / vol 100GB) vía GraphQL
  - El pod descarga worker.py de GitHub y lo ejecuta
  - Recibe callbacks HMAC del pod (progress/completed/error)
  - Watchdog cada 5 min: mata pods huérfanos o sin heartbeat
  - Cuando termina: TERMINA EL POD (jamás deja GPU cobrando)
  - Si falla: guarda log completo, descargable por endpoint

Endpoints:
  GET  /                              → Página web (testeador drag&drop)
  POST /api/jobs                      → recibe ZIP, alquila Pod, devuelve job_id
  GET  /api/jobs/{id}                 → estado del job
  GET  /api/jobs/{id}/download        → URL del .ply terminado
  GET  /api/jobs/{id}/log             → log del worker (texto plano descargable)
  POST /api/internal/callback/{id}    → callback HMAC del pod
  GET  /api/health                    → health check
"""

import os, uuid, json, hmac, hashlib, sqlite3, asyncio
from datetime import datetime, timezone
from contextlib import contextmanager

import boto3
from botocore.config import Config
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
import uvicorn

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════

RUNPOD_API_KEY      = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_API_URL      = "https://api.runpod.io/graphql"

# Imagen base PyTorch+CUDA compatible con gsplat 1.4.0
RUNPOD_IMAGE        = "runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04"

# Configuración del Pod (LO QUE PEDISTE: cont 50, vol 100)
POD_CONTAINER_DISK_GB = 50
POD_VOLUME_DISK_GB    = 100
POD_MIN_VCPU          = 8
POD_MIN_MEMORY_GB     = 32

# GPU preferida (RTX 4090) y fallbacks COMPATIBLES con nuestro stack
# Excluye Blackwell (RTX PRO 6000) que rompía PyTorch 2.1
# Ranking PURO POR RENDIMIENTO para 3D Gaussian Splatting.
# RunPod prueba en este orden; salta si una no está disponible.
# Excluye Blackwell (rompe PyTorch 2.1) y H100/H200 (overkill caro).
GPU_PREFERENCE_NAMES = [
    # ───── 1° lugar — la que pediste primero ─────
    "NVIDIA GeForce RTX 4090",            # 24GB · Ada · best perf/$ para 3DGS

    # ───── Premium Ada Lovelace (mismo arch que 4090, más VRAM) ─────
    "NVIDIA RTX 6000 Ada",     # 48GB · Ada · doble VRAM, ideal escenas grandes
    "NVIDIA L40S",                         # 48GB · Ada · datacenter
    "NVIDIA L40",                          # 48GB · Ada

    # ───── A100 (Ampere top, mucha VRAM) ─────
    "NVIDIA A100-SXM4-80GB",              # 80GB · Ampere · máxima banda ancha
    "NVIDIA A100 80GB PCIe",              # 80GB · Ampere
    "NVIDIA A100 80GB",                   # alias por si RunPod renombra
    "NVIDIA A100-PCIE-40GB",              # 40GB · Ampere

    # ───── Pro Ampere (48GB) ─────
    "NVIDIA RTX A6000",                   # 48GB · Ampere · gran VRAM, casi siempre disponible
    "NVIDIA A40",                          # 48GB · Ampere · datacenter equivalente

    # ───── Consumer Ampere 24GB ─────
    "NVIDIA GeForce RTX 3090 Ti",         # 24GB · Ampere top consumer
    "NVIDIA GeForce RTX 3090",            # 24GB · Ampere
    "NVIDIA RTX A5000",                   # 24GB · Ampere pro

    # ───── Ada recortadas / Ampere VRAM justa ─────
    "NVIDIA L4",                           # 24GB · Ada inference-oriented
    "NVIDIA RTX A4500",                   # 20GB · Ampere · VRAM justa
    "NVIDIA RTX A4000",                   # 16GB · Ampere · último recurso (solo fast)
]

# Configuración auto-adaptativa de disco según GPU
GPU_DISK_CONFIG = {
    # GPUs de 80GB VRAM (A100) → más disco para modelos cacheados
    "NVIDIA A100-SXM4-80GB":      {"container": 60, "volume": 120},
    "NVIDIA A100 80GB PCIe":      {"container": 60, "volume": 120},
    "NVIDIA A100 80GB":           {"container": 60, "volume": 120},
    "NVIDIA A100-PCIE-40GB":      {"container": 60, "volume": 120},
    # GPUs con VRAM justa → ahorramos disco también
    "NVIDIA RTX A4500":           {"container": 40, "volume": 80},
    "NVIDIA RTX A4000":           {"container": 40, "volume": 80},
    # El resto (24-48GB): configuración estándar que pediste (50/100)
}

# URL pública del worker.py en GitHub (lo que el pod baja al arrancar)
WORKER_SCRIPT_URL   = os.environ.get(
    "WORKER_SCRIPT_URL",
    "https://raw.githubusercontent.com/Felipegil0106/gaussian-worker/main/worker.py")

# R2
R2_ACCOUNT_ID  = os.environ.get("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY  = os.environ.get("R2_ACCESS_KEY", "")
R2_SECRET_KEY  = os.environ.get("R2_SECRET_KEY", "")
R2_BUCKET      = os.environ.get("R2_BUCKET", "gaussian-scanner")

# URL pública del backend (donde el pod manda callbacks)
BACKEND_URL    = os.environ.get("BACKEND_URL",
                                "https://gaussian-backend-production.up.railway.app")

# Secreto HMAC para firmar callbacks (debe ser el mismo del worker)
CALLBACK_SECRET = os.environ.get("CALLBACK_SECRET", "")
if not CALLBACK_SECRET:
    # Genera uno si falta — pero AVISA en logs
    CALLBACK_SECRET = uuid.uuid4().hex
    print("[WARN] CALLBACK_SECRET no estaba seteado, generado uno nuevo")

# Watchdog
POD_MAX_LIFETIME_MIN     = 90
POD_HEARTBEAT_TIMEOUT_MIN = 20
WATCHDOG_INTERVAL_SEC    = 300  # 5 min

PORT = int(os.environ.get("PORT", "8000"))
DB_PATH = os.environ.get("DB_PATH", "/data/jobs.db")

# ══════════════════════════════════════════════════════════════
# FASTAPI
# ══════════════════════════════════════════════════════════════

app = FastAPI(title="Gaussian Scanner API v3", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ══════════════════════════════════════════════════════════════
# BASE DE DATOS
# ══════════════════════════════════════════════════════════════

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT DEFAULT 'uploading',
                quality TEXT DEFAULT 'fast',
                pod_id TEXT,
                gpu_type TEXT,
                created_at TEXT,
                updated_at TEXT,
                last_heartbeat TEXT,
                progress REAL DEFAULT 0,
                message TEXT,
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
        # Migraciones idempotentes
        for col, typ in [("pod_id","TEXT"),("gpu_type","TEXT"),
                         ("last_heartbeat","TEXT"),("progress","REAL"),
                         ("message","TEXT"),("worker_log","TEXT")]:
            try: db.execute(f"ALTER TABLE jobs ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError: pass

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def job_update(job_id, **fields):
    fields["updated_at"] = datetime.now(timezone.utc).isoformat()
    cols = ", ".join(f"{k}=?" for k in fields.keys())
    vals = list(fields.values()) + [job_id]
    with get_db() as db:
        db.execute(f"UPDATE jobs SET {cols} WHERE id=?", vals)

def job_get(job_id):
    with get_db() as db:
        r = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(r) if r else None

# ══════════════════════════════════════════════════════════════
# R2 CLIENTE
# ══════════════════════════════════════════════════════════════

def get_r2():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

def r2_put_url(key, expires=7200):
    return get_r2().generate_presigned_url(
        "put_object", Params={"Bucket":R2_BUCKET,"Key":key}, ExpiresIn=expires)

def r2_get_url(key, expires=7200):
    return get_r2().generate_presigned_url(
        "get_object", Params={"Bucket":R2_BUCKET,"Key":key}, ExpiresIn=expires)

def r2_upload_file(file_obj, key):
    get_r2().upload_fileobj(file_obj, R2_BUCKET, key)

# ══════════════════════════════════════════════════════════════
# RUNPOD GRAPHQL (orquestación de Pod)
# ══════════════════════════════════════════════════════════════

class RunPod:

    @staticmethod
    async def _query(query, variables=None):
        headers = {"Content-Type":"application/json",
                   "Authorization":f"Bearer {RUNPOD_API_KEY}"}
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(RUNPOD_API_URL, headers=headers,
                            json={"query":query, "variables":variables or {}})
            r.raise_for_status()
            data = r.json()
            if "errors" in data:
                raise RuntimeError(f"RunPod GraphQL: {data['errors']}")
            return data["data"]

    @staticmethod
    async def find_best_gpu():
        """Devuelve (gpu_id, gpu_name) según el ranking de preferencia.
        El orden es por RENDIMIENTO; RunPod salta automáticamente las no disponibles
        al intentar el podFindAndDeployOnDemand."""
        try:
            data = await RunPod._query("""
                query { gpuTypes { id displayName memoryInGb } }
            """)
        except Exception as e:
            print(f"[runpod] find_best_gpu error: {e}")
            return None, None

        all_gpus = data.get("gpuTypes", [])
        by_name = {g["displayName"]: g["id"] for g in all_gpus}

        # Recorrer la lista preferida EN ORDEN — RunPod usará la primera que exista
        # y que tenga capacidad. Si la #1 no hay, el caller reintenta con la #2, etc.
        for pref in GPU_PREFERENCE_NAMES:
            if pref in by_name:
                print(f"[runpod] GPU intentada: {pref}")
                return by_name[pref], pref
        # Último fallback: cualquier GPU con ≥24 GB que NO sea Blackwell
        BLACKWELL_BANNED = ("RTX 5090", "RTX PRO 6000", "RTX PRO 4500",
                            "RTX PRO 4000", "B200", "B300")
        for g in all_gpus:
            name = g.get("displayName", "")
            if g.get("memoryInGb", 0) >= 24 and not any(b in name for b in BLACKWELL_BANNED):
                print(f"[runpod] GPU fallback genérico: {name}")
                return g["id"], name
        return None, None

    @staticmethod
    def disk_config_for(gpu_name):
        """Config óptima de disco según GPU asignada."""
        return GPU_DISK_CONFIG.get(gpu_name, {
            "container": POD_CONTAINER_DISK_GB,
            "volume": POD_VOLUME_DISK_GB,
        })

    @staticmethod
    async def try_create_pod_with_fallbacks(job_id, env_vars):
        """Intenta crear el pod recorriendo el ranking de GPUs.
        Si una falla con SUPPLY_CONSTRAINT, prueba la siguiente automáticamente.
        Devuelve (pod, gpu_name, disk) o lanza excepción si todas fallaron."""
        try:
            data = await RunPod._query("""
                query { gpuTypes { id displayName memoryInGb } }
            """)
        except Exception as e:
            raise RuntimeError(f"No pude listar GPUs de RunPod: {e}")
        all_gpus = data.get("gpuTypes", [])
        by_name = {g["displayName"]: g["id"] for g in all_gpus}

        errors = []
        for pref in GPU_PREFERENCE_NAMES:
            gpu_id = by_name.get(pref)
            if not gpu_id:
                continue
            disk = RunPod.disk_config_for(pref)
            print(f"[fallback] Intentando {pref} (cont={disk['container']} vol={disk['volume']})")
            try:
                pod = await RunPod.create_pod(
                    job_id, gpu_id, env_vars,
                    container_gb=disk["container"],
                    volume_gb=disk["volume"],
                )
                print(f"[fallback] ✓ {pref} alquilada exitosamente")
                return pod, pref, disk
            except Exception as e:
                msg = str(e)
                # Si es supply constraint, probar la siguiente del ranking
                if "SUPPLY_CONSTRAINT" in msg or "no longer any instances" in msg.lower():
                    print(f"[fallback] {pref} sin disponibilidad, probando siguiente...")
                    errors.append(f"{pref}: sin stock")
                    continue
                # Otros errores: re-lanzar
                raise
        raise RuntimeError(
            f"Ninguna GPU del ranking está disponible ahora. "
            f"Intentos: {'; '.join(errors[:5])}. Reintenta en 5-10 min."
        )

    @staticmethod
    async def create_pod(job_id, gpu_type_id, env_vars, container_gb=None, volume_gb=None):
        """Crea un Pod on-demand. Bootstrap descarga worker.py y lo ejecuta."""
        container_gb = container_gb or POD_CONTAINER_DISK_GB
        volume_gb = volume_gb or POD_VOLUME_DISK_GB
        # Bootstrap: instala COLMAP, gsplat, deps, baja worker.py, lo corre
        bootstrap = (
            "bash -lc 'set -e; "
            "echo \"[bootstrap] iniciando\"; "
            "apt-get update -qq; "
            "apt-get install -y -qq git wget ffmpeg colmap python3-pip "
            "  libgl1-mesa-glx libglib2.0-0 nodejs npm; "
            "echo \"[bootstrap] sistema OK\"; "
            "pip install -q --upgrade pip; "
            "pip install -q boto3 plyfile opencv-python-headless requests tqdm numpy pillow; "
            "pip install -q transformers accelerate timm safetensors huggingface_hub; "
            "echo \"[bootstrap] python deps OK\"; "
            # gsplat v1.4.0 — la versión que SÍ funciona (sin bug color_correct)
            "git clone --branch v1.4.0 --depth 1 "
            "  https://github.com/nerfstudio-project/gsplat.git /opt/gsplat-repo; "
            "cd /opt/gsplat-repo && pip install -q . && "
            "  pip install -q -r examples/requirements.txt; "
            "echo \"[bootstrap] gsplat v1.4.0 OK\"; "
            "npm install -g @playcanvas/splat-transform 2>/dev/null || true; "
            f"wget -q -O /workspace/worker.py \"{WORKER_SCRIPT_URL}\"; "
            "echo \"[bootstrap] worker descargado, ejecutando\"; "
            "cd /workspace && python3 -u worker.py'"
        )
        env_list = [{"key":k, "value":str(v)} for k, v in env_vars.items()]
        mutation = """
        mutation deploy($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id machineId desiredStatus
            }
        }
        """
        variables = {"input": {
            "cloudType":"SECURE",
            "gpuCount":1,
            "volumeInGb": volume_gb,            # auto-ajustado a la GPU
            "containerDiskInGb": container_gb,  # auto-ajustado a la GPU
            "minVcpuCount": POD_MIN_VCPU,
            "minMemoryInGb": POD_MIN_MEMORY_GB,
            "gpuTypeId": gpu_type_id,
            "name": f"gaussian-{job_id[:8]}",
            "imageName": RUNPOD_IMAGE,
            "dockerArgs": bootstrap,
            "ports":"8888/http",
            "volumeMountPath":"/workspace",
            "env": env_list,
        }}
        print(f"[runpod] Creando pod gpu={gpu_type_id} cont={container_gb}GB vol={volume_gb}GB")
        data = await RunPod._query(mutation, variables)
        return data["podFindAndDeployOnDemand"]

    @staticmethod
    async def terminate_pod(pod_id):
        """Destruye el Pod definitivamente. Deja de cobrar."""
        if not pod_id: return False
        try:
            await RunPod._query("""
                mutation t($input: PodTerminateInput!) {
                    podTerminate(input: $input)
                }
            """, {"input":{"podId":pod_id}})
            print(f"[runpod] Pod {pod_id} terminado")
            return True
        except Exception as e:
            print(f"[runpod] terminate falló: {e}")
            return False

    @staticmethod
    async def list_my_pods():
        try:
            data = await RunPod._query("""
                query { myself { pods { id name desiredStatus runtime { uptimeInSeconds } } } }
            """)
            return data.get("myself", {}).get("pods", []) or []
        except Exception:
            return []

# ══════════════════════════════════════════════════════════════
# HMAC
# ══════════════════════════════════════════════════════════════

def verify_signature(body: bytes, sig: str) -> bool:
    if not sig: return False
    expected = hmac.new(CALLBACK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)

# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE

@app.get("/api/health")
def health():
    return {"status":"ok", "time":datetime.now(timezone.utc).isoformat(),
            "runpod_configured": bool(RUNPOD_API_KEY),
            "r2_configured": bool(R2_ACCOUNT_ID and R2_ACCESS_KEY)}

@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), quality: str = Form("fast")):
    if quality not in ("fast","balanced","quality"):
        raise HTTPException(400, "quality debe ser: fast, balanced, quality")

    job_id = str(uuid.uuid4())[:12]
    now = datetime.now(timezone.utc).isoformat()
    zip_key = f"uploads/{job_id}/input.zip"
    ply_key = f"results/{job_id}/scene.ply"
    glb_key = f"results/{job_id}/collision.glb"

    with get_db() as db:
        db.execute("""
            INSERT INTO jobs (id,status,quality,created_at,updated_at,ply_key,glb_key,
                              last_heartbeat,progress,message)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (job_id,"uploading",quality,now,now,ply_key,glb_key,now,0.0,"Subiendo a R2"))

    # Subir ZIP a R2
    try:
        r2_upload_file(file.file, zip_key)
    except Exception as e:
        job_update(job_id, status="error", error=f"R2 upload falló: {e}")
        raise HTTPException(500, f"R2 falló: {e}")

    # Construir env vars del pod
    env = {
        "TOUR_ID": job_id,
        "INPUT_URL": r2_get_url(zip_key, expires=7200),
        "UPLOAD_URL_PLY": r2_put_url(ply_key, expires=7200),
        "UPLOAD_URL_GLB": r2_put_url(glb_key, expires=7200),
        "CALLBACK_URL": f"{BACKEND_URL}/api/internal/callback/{job_id}",
        "CALLBACK_SECRET": CALLBACK_SECRET,
        "QUALITY": quality,
    }

    # Recorrer el ranking de GPUs hasta encontrar una con stock
    try:
        pod, gpu_name, disk = await RunPod.try_create_pod_with_fallbacks(job_id, env)
    except Exception as e:
        job_update(job_id, status="error", error=f"Crear Pod falló: {e}")
        raise HTTPException(503, f"Sin GPU disponible: {e}")

    job_update(job_id, status="processing",
               pod_id=pod.get("id",""),
               gpu_type=gpu_name,
               message=f"Pod {gpu_name} arrancando (~5 min bootstrap)")

    return {"job_id":job_id, "status":"processing", "quality":quality,
            "pod_id":pod.get("id"), "gpu_type":gpu_name,
            "container_gb":disk["container"], "volume_gb":disk["volume"]}

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    j = job_get(job_id)
    if not j:
        raise HTTPException(404, "Job no encontrado")
    return {
        "job_id": j["id"],
        "status": j["status"],
        "quality": j.get("quality"),
        "progress": j.get("progress") or 0,
        "message": j.get("message") or "",
        "pod_id": j.get("pod_id"),
        "gpu_type": j.get("gpu_type"),
        "frames_used": j.get("frames_used"),
        "ply_mb": j.get("ply_mb"),
        "has_collision": bool(j.get("has_collision")),
        "seconds": j.get("seconds"),
        "error": j.get("error"),
        "has_log": bool(j.get("worker_log")),
    }

@app.get("/api/jobs/{job_id}/download")
def download_result(job_id: str):
    j = job_get(job_id)
    if not j: raise HTTPException(404, "Job no encontrado")
    if j["status"] != "completed":
        raise HTTPException(400, f"Job no listo (estado: {j['status']})")
    result = {"job_id":job_id, "ply_url":r2_get_url(j["ply_key"]),
              "ply_mb":j.get("ply_mb",0)}
    if j.get("has_collision"):
        result["glb_url"] = r2_get_url(j["glb_key"])
    return result

@app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
def get_log(job_id: str):
    j = job_get(job_id)
    if not j: raise HTTPException(404, "Job no encontrado")
    header = (
        f"================================================\n"
        f"LOG DE RENDERIZADO\n"
        f"Job: {job_id}\n"
        f"Estado: {j['status']}\n"
        f"Calidad: {j.get('quality')}\n"
        f"GPU: {j.get('gpu_type')}\n"
        f"Pod ID: {j.get('pod_id')}\n"
        f"Error: {j.get('error') or 'N/A'}\n"
        f"================================================\n\n"
    )
    return header + (j.get("worker_log") or "(Sin log)")

@app.post("/api/internal/callback/{job_id}")
async def worker_callback(job_id: str, request: Request,
                          x_signature: str = Header(default="")):
    body = await request.body()
    if not verify_signature(body, x_signature):
        raise HTTPException(401, "Firma HMAC inválida")
    try:
        payload = json.loads(body)
    except Exception:
        raise HTTPException(400, "JSON inválido")

    j = job_get(job_id)
    if not j: raise HTTPException(404, "Job no encontrado")

    cb_type = payload.get("type", "")
    now = datetime.now(timezone.utc).isoformat()

    if cb_type == "progress":
        job_update(job_id,
                   progress=float(payload.get("progress", 0)),
                   message=payload.get("message", "")[:200],
                   last_heartbeat=now)
        return {"ok":True}

    elif cb_type == "completed":
        job_update(job_id,
                   status="completed",
                   progress=1.0,
                   message="Completado",
                   frames_used=payload.get("frames_used", 0),
                   ply_mb=payload.get("ply_mb", 0),
                   has_collision=1 if payload.get("has_collision") else 0,
                   seconds=payload.get("seconds", 0),
                   last_heartbeat=now)
        # TERMINAR EL POD (clave: no dejar GPU cobrando)
        if j.get("pod_id"):
            await RunPod.terminate_pod(j["pod_id"])
        return {"ok":True}

    elif cb_type == "error":
        log_text = payload.get("log", "") or payload.get("error_message", "")
        job_update(job_id,
                   status="error",
                   error=payload.get("error_message","Error desconocido")[:500],
                   worker_log=log_text,
                   last_heartbeat=now)
        if j.get("pod_id"):
            await RunPod.terminate_pod(j["pod_id"])
        return {"ok":True}

    return {"ok":False, "reason":"tipo callback desconocido"}

# ══════════════════════════════════════════════════════════════
# WATCHDOG (mata pods huérfanos o sin heartbeat)
# ══════════════════════════════════════════════════════════════

async def watchdog_loop():
    while True:
        try:
            await _watchdog_pass()
        except Exception as e:
            print(f"[watchdog] error: {e}")
        await asyncio.sleep(WATCHDOG_INTERVAL_SEC)

async def _watchdog_pass():
    now = datetime.now(timezone.utc)
    # 1) Jobs procesando con heartbeat viejo → matar pod
    with get_db() as db:
        rows = db.execute("""
            SELECT id, pod_id, last_heartbeat FROM jobs
            WHERE status='processing' AND pod_id IS NOT NULL AND pod_id != ''
        """).fetchall()
    for r in rows:
        hb = r["last_heartbeat"]
        if not hb: continue
        try:
            hb_dt = datetime.fromisoformat(hb.replace("Z","+00:00"))
            age_min = (now - hb_dt).total_seconds() / 60
        except Exception:
            continue
        if age_min > POD_HEARTBEAT_TIMEOUT_MIN:
            print(f"[watchdog] Job {r['id']} sin HB hace {age_min:.0f} min — matando pod")
            await RunPod.terminate_pod(r["pod_id"])
            job_update(r["id"], status="error",
                       error=f"Sin heartbeat hace {age_min:.0f} min (timeout)")

    # 2) Pods de RunPod que NO están en nuestra DB → huérfanos, matar
    try:
        pods = await RunPod.list_my_pods()
    except Exception:
        pods = []
    known_pods = set()
    with get_db() as db:
        for r in db.execute("SELECT pod_id FROM jobs WHERE pod_id IS NOT NULL").fetchall():
            if r["pod_id"]:
                known_pods.add(r["pod_id"])

    for p in pods:
        pid = p.get("id")
        if pid and pid not in known_pods:
            uptime = (p.get("runtime") or {}).get("uptimeInSeconds", 0)
            if uptime and uptime > POD_MAX_LIFETIME_MIN * 60:
                print(f"[watchdog] Pod huérfano {pid} con {uptime}s — terminando")
                await RunPod.terminate_pod(pid)

# ══════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    init_db()
    asyncio.create_task(watchdog_loop())
    print(f"Backend v3 iniciado. RunPod={'OK' if RUNPOD_API_KEY else 'NO'}, "
          f"R2={'OK' if R2_ACCOUNT_ID else 'NO'}, BACKEND_URL={BACKEND_URL}")

# ══════════════════════════════════════════════════════════════
# HTML — la página drag&drop (misma que ya tenías + adaptada)
# ══════════════════════════════════════════════════════════════

HTML_PAGE = """<!DOCTYPE html>
<html lang="es"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gaussian Scanner — Test Render</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:linear-gradient(135deg,#0f0f0f 0%,#1a1a2e 100%);color:#eee;min-height:100vh;padding:20px}
.container{max-width:720px;margin:0 auto}
h1{text-align:center;font-size:28px;margin-bottom:8px;
  background:linear-gradient(90deg,#FF6B35,#00D9FF);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.subtitle{text-align:center;color:#888;margin-bottom:24px;font-size:14px}
.card{background:#1a1a1a;border-radius:16px;padding:24px;margin-bottom:20px;border:1px solid #2a2a2a}
#dropzone{border:2px dashed #FF6B35;border-radius:12px;padding:48px 24px;text-align:center;
  cursor:pointer;transition:.2s;background:#1f1f1f}
#dropzone:hover,#dropzone.drag{background:#2a2018;border-color:#00D9FF}
.icon{font-size:48px;margin-bottom:12px}
.text{font-size:16px;color:#ccc}
.sub{font-size:13px;color:#777;margin-top:8px}
.file-info{background:#0f2a0f;border:1px solid #2a5a2a;border-radius:8px;padding:12px;margin-top:16px;display:none}
select,button{width:100%;padding:14px;border-radius:10px;border:none;font-size:15px;margin-top:12px;cursor:pointer}
select{background:#2a2a2a;color:#eee}
.btn-primary{background:linear-gradient(90deg,#FF6B35,#C84B1A);color:#fff;font-weight:bold;font-size:16px}
.btn-primary:disabled{opacity:.4;cursor:not-allowed}
.btn-download{background:#00D9FF;color:#000;font-weight:bold}
.btn-success{background:#4CAF50;color:#fff;font-weight:bold}
#progress{display:none}
.log-box{background:#0a0a0a;border:1px solid #2a2a2a;border-radius:8px;padding:16px;
  font-family:monospace;font-size:12px;color:#9fef9f;height:280px;overflow-y:auto;
  white-space:pre-wrap;margin-top:16px}
.status{text-align:center;padding:16px;font-size:16px;font-weight:bold}
.status.processing{color:#00D9FF}.status.success{color:#4CAF50}.status.error{color:#E53935}
.spinner{display:inline-block;width:16px;height:16px;border:3px solid #333;border-top-color:#00D9FF;
  border-radius:50%;animation:spin 1s linear infinite;vertical-align:middle;margin-right:8px}
@keyframes spin{to{transform:rotate(360deg)}}
.hidden{display:none !important}
.bar{height:4px;background:#222;border-radius:2px;margin-top:12px;overflow:hidden}
.bar-fill{height:100%;background:linear-gradient(90deg,#FF6B35,#00D9FF);transition:width .3s}
</style></head><body>
<div class="container">
  <h1>🎨 Gaussian Scanner — Test Render</h1>
  <p class="subtitle">Pod RTX 4090 on-demand · 50GB cont / 100GB vol · auto-destruye al terminar</p>
  <div class="card" id="upload-card">
    <div id="dropzone">
      <div class="icon">📦</div>
      <div class="text">Arrastra tu ZIP aquí o haz click</div>
      <div class="sub">.zip con fotos (mínimo 20)</div>
    </div>
    <input type="file" id="fileInput" accept=".zip" style="display:none">
    <div class="file-info" id="fileInfo"></div>
    <select id="quality">
      <option value="fast">Rápido (7K iter, ~10 min training)</option>
      <option value="balanced">Balanceado (30K iter, ~25 min)</option>
      <option value="quality">Máxima (50K iter, ~45 min)</option>
    </select>
    <button class="btn-primary" id="renderBtn" disabled>🚀 Iniciar Renderizado</button>
  </div>
  <div class="card" id="progress">
    <div class="status processing" id="statusText"><span class="spinner"></span>Procesando...</div>
    <div class="bar"><div class="bar-fill" id="barFill" style="width:0%"></div></div>
    <div class="log-box" id="logBox">Iniciando...</div>
    <div id="resultActions" class="hidden">
      <button class="btn-success hidden" id="viewBtn">🎨 Descargar .ply</button>
      <button class="btn-download hidden" id="logBtn">📄 Descargar log del error</button>
      <button class="btn-primary" id="newBtn">🔄 Probar otro ZIP</button>
    </div>
  </div>
</div>
<script>
const dz=document.getElementById('dropzone'),fi=document.getElementById('fileInput');
const info=document.getElementById('fileInfo'),btn=document.getElementById('renderBtn');
const qsel=document.getElementById('quality'),up=document.getElementById('upload-card');
const pr=document.getElementById('progress'),st=document.getElementById('statusText');
const lb=document.getElementById('logBox'),ra=document.getElementById('resultActions');
const vb=document.getElementById('viewBtn'),lgb=document.getElementById('logBtn');
const nb=document.getElementById('newBtn'),bf=document.getElementById('barFill');
let sel=null,jid=null,timer=null;
dz.onclick=()=>fi.click();
dz.ondragover=e=>{e.preventDefault();dz.classList.add('drag')};
dz.ondragleave=()=>dz.classList.remove('drag');
dz.ondrop=e=>{e.preventDefault();dz.classList.remove('drag');if(e.dataTransfer.files.length)hf(e.dataTransfer.files[0])};
fi.onchange=e=>{if(e.target.files.length)hf(e.target.files[0])};
function hf(f){if(!f.name.toLowerCase().endsWith('.zip')){alert('Sube un .zip');return}
  sel=f;info.style.display='block';info.textContent='✓ '+f.name+' ('+(f.size/1048576).toFixed(1)+' MB)';btn.disabled=false}
function addLog(m){lb.textContent+='\\n'+m;lb.scrollTop=lb.scrollHeight}
btn.onclick=async()=>{if(!sel)return;up.classList.add('hidden');pr.style.display='block';
  ra.classList.add('hidden');lb.textContent='Subiendo ZIP a R2...';st.innerHTML='<span class="spinner"></span>Subiendo...';
  const fd=new FormData();fd.append('file',sel);fd.append('quality',qsel.value);
  try{const r=await fetch('/api/jobs',{method:'POST',body:fd});
    if(!r.ok){throw new Error('HTTP '+r.status+': '+await r.text())}
    const d=await r.json();jid=d.job_id;
    addLog('✓ Job '+jid+' creado');addLog('✓ Pod RTX 4090 alquilado (gpu='+d.gpu_type+')');
    addLog('Esperando bootstrap del pod (~5 min)... luego renderizado');
    st.innerHTML='<span class="spinner"></span>Pod arrancando...';startPoll()
  }catch(e){addLog('❌ '+e.message);st.className='status error';st.textContent='❌ Error';showNew()}};
function startPoll(){let el=0;
  timer=setInterval(async()=>{el+=10;
    try{const r=await fetch('/api/jobs/'+jid);const j=await r.json();
      const p=Math.round((j.progress||0)*100);bf.style.width=p+'%';
      st.innerHTML='<span class="spinner"></span>'+(j.message||'Procesando')+' ('+p+'%)';
      if(j.status==='completed'){clearInterval(timer);
        addLog('');addLog('✅ RENDER COMPLETADO');
        addLog('Frames: '+(j.frames_used||'?')+' · '+(j.ply_mb||'?')+' MB · '+(j.seconds||'?')+'s');
        st.className='status success';st.textContent='✅ ¡Completado!';bf.style.width='100%';
        ra.classList.remove('hidden');vb.classList.remove('hidden');lgb.classList.add('hidden');
        vb.onclick=async()=>{const dr=await fetch('/api/jobs/'+jid+'/download');const dd=await dr.json();
          window.open(dd.ply_url,'_blank')};
        showNew()
      }else if(j.status==='error'){clearInterval(timer);
        addLog('');addLog('❌ ERROR: '+(j.error||'sin detalle'));
        st.className='status error';st.textContent='❌ Falló';
        ra.classList.remove('hidden');vb.classList.add('hidden');
        if(j.has_log){lgb.classList.remove('hidden');
          lgb.onclick=()=>window.open('/api/jobs/'+jid+'/log','_blank')}
        showNew()
      }
    }catch(e){addLog('⚠ '+e.message)}
  },10000)}
function showNew(){nb.onclick=()=>location.reload()}
</script></body></html>"""

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
