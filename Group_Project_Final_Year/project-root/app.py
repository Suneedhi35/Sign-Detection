"""
AI-Enabled Indian Sign Language Communication System
====================================================
Features:
  - Secure login + user registration (SQLite-backed, SHA-256 hashed passwords)
  - Live camera detection  →  YOLOv8 + ViT classification
  - ISL Dictionary browser (from CSV)
  - Multilingual translation (15+ Indian + global languages)
  - Text-to-Speech via gTTS (supports all Indian languages)
  - Speech-to-Text via OpenAI Whisper (supports 90+ languages)
  - Auto-pipeline: Camera → Translation → TTS / STT
  - Admin panel: view and manage registered users
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import ViTForImageClassification
from ultralytics import YOLO
from deep_translator import GoogleTranslator
import pandas as pd
import tempfile
import os
import io
import time
import hashlib
import sqlite3
import re
from datetime import datetime

# ──────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sign Language Recognition and Communication System",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
#  DATABASE  (SQLite — persists in project root as users.db)
# ──────────────────────────────────────────────────────────────
_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")


def _get_conn():
    """Return a thread-safe SQLite connection."""
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ──────────────────────────────────────────────────────────────
#  AUTHENTICATION HELPERS
# ──────────────────────────────────────────────────────────────
def _hash(pw: str) -> str:
    """SHA-256 hash a password string."""
    return hashlib.sha256(pw.encode()).hexdigest()


def _init_db():
    """Create the users table and seed default accounts if absent."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                username     TEXT    NOT NULL UNIQUE,
                password_hash TEXT   NOT NULL,
                display_name TEXT    NOT NULL,
                email        TEXT    NOT NULL DEFAULT '',
                role         TEXT    NOT NULL DEFAULT 'user',
                is_approved  INTEGER NOT NULL DEFAULT 1,
                created_at   TEXT    NOT NULL,
                last_login   TEXT
            )
        """)
        # Seed default accounts on first run
        for uname, pw, role, display in [
            ("admin", "isl@Admin2024", "admin", "Administrator"),
            ("user",  "isl@User2024",  "user",  "Guest User"),
        ]:
            conn.execute("""
                INSERT OR IGNORE INTO users
                    (username, password_hash, display_name, email, role, is_approved, created_at)
                VALUES (?, ?, ?, ?, ?, 1, ?)
            """, (uname, _hash(pw), display, "", role,
                  datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()


def _authenticate(username: str, password: str):
    """Return user row dict or None."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username=? AND is_approved=1",
            (username.strip().lower(),)
        ).fetchone()
    if row and row["password_hash"] == _hash(password):
        # Update last_login timestamp
        with _get_conn() as conn:
            conn.execute(
                "UPDATE users SET last_login=? WHERE username=?",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 username.strip().lower())
            )
            conn.commit()
        return dict(row)
    return None


def _register_user(username: str, password: str, display_name: str, email: str) -> tuple[bool, str]:
    """Register a new user. Returns (success, message)."""
    username = username.strip().lower()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if not re.match(r'^[a-z0-9_]+$', username):
        return False, "Username may only contain letters, numbers, and underscores."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if email and not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
        return False, "Please enter a valid email address."
    try:
        with _get_conn() as conn:
            conn.execute("""
                INSERT INTO users
                    (username, password_hash, display_name, email, role, is_approved, created_at)
                VALUES (?, ?, ?, ?, 'user', 1, ?)
            """, (username, _hash(password),
                  display_name.strip() or username,
                  email.strip(),
                  datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
        return True, "Account created successfully! You can now log in."
    except sqlite3.IntegrityError:
        return False, "Username already taken. Please choose another."


# Initialise DB on startup
_init_db()


# ──────────────────────────────────────────────────────────────
#  AUTH UI  (Login + Register tabs)
# ──────────────────────────────────────────────────────────────
def _auth_ui():
    """Render the combined Login / Register screen."""
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        # Header banner
        st.markdown("""
        <div style='background:linear-gradient(135deg,#667eea,#764ba2);
                    padding:2rem 2rem 1.5rem;border-radius:18px;
                    box-shadow:0 12px 40px rgba(102,126,234,.45);
                    margin-top:50px;text-align:center'>
          <h2 style='color:white;margin:0 0 .3rem 0;font-size:2.2rem'>🤟 Sign Language Recognition</h2>
          <p style='color:rgba(255,255,255,.8);margin:0'>
            Sign Language Recognition and Communication System</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        tab_login, tab_reg = st.tabs(["🔑 Login", "📝 Register"])

        # ── LOGIN TAB ──────────────────────────────────────────
        with tab_login:
            with st.form("login_form"):
                username = st.text_input("👤 Username", placeholder="Enter username",
                                         key="login_user")
                password = st.text_input("🔒 Password", type="password",
                                         placeholder="Enter password", key="login_pw")
                submitted = st.form_submit_button("🚀 Login", use_container_width=True)

            if submitted:
                rec = _authenticate(username, password)
                if rec:
                    st.session_state.update({
                        "authenticated": True,
                        "username":      rec["username"],
                        "user_role":     rec["role"],
                        "user_display":  rec["display_name"],
                        "user_email":    rec["email"],
                    })
                    st.rerun()
                else:
                    with _get_conn() as conn:
                        row = conn.execute(
                            "SELECT is_approved FROM users WHERE username=?",
                            (username.strip().lower(),)
                        ).fetchone()
                    if row and not row["is_approved"]:
                        st.warning("⏳ Your account is awaiting admin approval.")
                    else:
                        st.error("❌ Invalid username or password.")

            st.markdown("""
            <div style='text-align:center;margin-top:.6rem;
                        color:#888;font-size:.8rem'>
              Default — <b>admin</b> / <b>isl@Admin2024</b>
            </div>""", unsafe_allow_html=True)

        # ── REGISTER TAB ───────────────────────────────────────
        with tab_reg:
            st.markdown("""
            <div style='background:rgba(102,126,234,.08);border-radius:10px;
                        padding:.8rem 1rem;margin-bottom:.8rem;
                        border-left:4px solid #667eea;font-size:.88rem;color:#555'>
              Create a free account to access all ISL features.
            </div>""", unsafe_allow_html=True)

            with st.form("register_form"):
                r_user    = st.text_input("👤 Username *",
                                          placeholder="Only letters, numbers, _",
                                          key="reg_user")
                r_display = st.text_input("🏷️ Display Name",
                                          placeholder="Your full name (optional)",
                                          key="reg_display")
                r_email   = st.text_input("📧 Email",
                                          placeholder="you@example.com (optional)",
                                          key="reg_email")
                r_pw      = st.text_input("🔒 Password *", type="password",
                                          placeholder="Min. 6 characters",
                                          key="reg_pw")
                r_pw2     = st.text_input("🔒 Confirm Password *", type="password",
                                          placeholder="Repeat password",
                                          key="reg_pw2")
                reg_btn   = st.form_submit_button("✅ Create Account",
                                                   use_container_width=True)

            if reg_btn:
                if not r_user or not r_pw or not r_pw2:
                    st.error("❌ Username and password fields are required.")
                elif r_pw != r_pw2:
                    st.error("❌ Passwords do not match.")
                else:
                    ok, msg = _register_user(r_user, r_pw, r_display, r_email)
                    if ok:
                        st.success(f"🎉 {msg}")
                    else:
                        st.error(f"❌ {msg}")


# ── Gate: show auth screen if not authenticated ──────────────
if not st.session_state.get("authenticated"):
    _auth_ui()
    st.stop()

# ──────────────────────────────────────────────────────────────
#  GLOBAL STYLES
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
h1 {
  background: linear-gradient(135deg,#667eea,#764ba2);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  font-size: 2.7rem !important; font-weight: 900 !important;
  text-align: center !important;
}
h2 { color: #667eea; border-bottom: 3px solid #764ba2; padding-bottom: .4rem; }
.stButton>button {
  background: linear-gradient(135deg,#667eea,#764ba2) !important;
  color: white !important; font-weight: bold !important;
  border: none !important; border-radius: 10px !important;
  padding: 10px 24px !important; transition: transform .2s !important;
}
.stButton>button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 16px rgba(102,126,234,.4) !important;
}
.card {
  background: linear-gradient(135deg,#667eea,#764ba2);
  padding: 1.2rem; border-radius: 12px; color: white; margin-bottom: 1rem;
}
.card h3 { margin: 0; color: white; }
.card-pink {
  background: linear-gradient(135deg,#f093fb,#f5576c);
  padding: 1.2rem; border-radius: 12px; color: white; margin-bottom: 1rem;
}
.success-box {
  background: linear-gradient(135deg,#84fab0,#8fd3f4);
  padding: 1.2rem; border-radius: 10px; color: #1a1a2e;
  font-weight: bold; text-align: center; margin: .8rem 0;
}
.info-box {
  background: linear-gradient(135deg,#fa709a,#fee140);
  padding: 1.2rem; border-radius: 10px; color: #1a1a2e;
  font-weight: bold; text-align: center; margin: .8rem 0;
}
.divider { border-top: 3px solid rgba(102,126,234,.3); margin: 2rem 0; }
.sidebar-user {
  background: linear-gradient(135deg,#667eea,#764ba2);
  padding: 1rem; border-radius: 10px; color: white;
  text-align: center; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-user">
      <b>👤 {st.session_state['user_display']}</b><br>
      <small>Role: {st.session_state['user_role']}</small>
    </div>""", unsafe_allow_html=True)

    if st.button("🚪 Logout", use_container_width=True):
        for k in ["authenticated", "username", "user_role", "user_display",
                  "running", "recognized_signs", "detected_word",
                  "last_translation", "last_trans_lang"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.markdown("---")
    st.markdown("### 📚 About")
    st.markdown("""
    **Models used**
    - 🔲 YOLOv8 – bounding-box hand detection
    - 🧠 ViT – sign classification (A–Z, 1–9)
    - 🎤 Whisper – multilingual STT
    - 🔊 gTTS – multilingual TTS

    **Languages**
    English · Hindi · Telugu · Tamil · Kannada ·
    Malayalam · Bengali · Marathi · Gujarati · Punjabi ·
    Assamese · Urdu · Odia · Sanskrit · Arabic
    """)

# ──────────────────────────────────────────────────────────────
#  LANGUAGE CONFIG
# ──────────────────────────────────────────────────────────────
LANGUAGES = {
    "English":   "en",
    "Hindi":     "hi",
    "Telugu":    "te",
    "Tamil":     "ta",
    "Kannada":   "kn",
    "Malayalam": "ml",
    "Bengali":   "bn",
    "Marathi":   "mr",
    "Gujarati":  "gu",
    "Punjabi":   "pa",
    "Assamese":  "as",
    "Urdu":      "ur",
    "Odia":      "or",
    "Sanskrit":  "sa",
    "Arabic":    "ar",
}

WHISPER_LANGS = {
    k: v for k, v in LANGUAGES.items()
    if k not in ("Sanskrit",)   # Whisper has no Sanskrit model
}

# ──────────────────────────────────────────────────────────────
#  MODEL PATHS
# ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(
    BASE_DIR,
    "AI-Enabled-Sign-Language-Communication-System-main",
    "models",
)

CLASS_NAMES = [
    "1","2","3","4","5","6","7","8","9",
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ──────────────────────────────────────────────────────────────
#  CACHED MODEL LOADERS
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Loading YOLOv8 model…")
def load_yolo():
    path = os.path.join(MODELS_DIR, "ISL-YOLOv8mBoundingBox.pt")
    return YOLO(path)


@st.cache_resource(show_spinner="⚙️ Loading ViT classifier…")
def load_vit():
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    path  = os.path.join(MODELS_DIR, "ISL-ViTImageClassification.pth")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


@st.cache_resource(show_spinner="⚙️ Loading Whisper STT model…")
def load_whisper():
    import whisper
    # Use "small" for better accuracy; "base" for speed
    return whisper.load_model("base")


@st.cache_resource(show_spinner="⚙️ Loading Word Recognition model…")
def load_word_predictor():
    """Load the trained LSTM word predictor (returns None if model not trained yet)."""
    try:
        from word_recognition.predict import WordPredictor
        predictor = WordPredictor()
        if predictor.is_loaded:
            return predictor
    except Exception as e:
        print(f"Word predictor not available: {e}")
    return None


# ──────────────────────────────────────────────────────────────
#  CSV DICTIONARY LOADER
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_dictionary():
    csv_path = os.path.join(BASE_DIR, "ISL_Dictionary_words.csv")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


isl_df = load_dictionary()
letter_links = {
    row["Folder Name"].strip(): row["Link"].strip()
    for _, row in isl_df.iterrows()
    if row["Folder Name"].strip() not in ["All Dictionary Videos", ""]
}

# ──────────────────────────────────────────────────────────────
#  INFERENCE HELPERS
# ──────────────────────────────────────────────────────────────
def detect_and_crop(frame: np.ndarray, yolo):
    results  = yolo(frame)
    annotated = frame.copy()
    crops     = []
    for box in results[0].boxes:
        if box.conf.item() >= 0.50:
            x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            crops.append(frame[y1:y2, x1:x2])
    return annotated, crops


def classify_sign(pil_img: Image.Image, vit) -> tuple[str, float]:
    t = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = vit(t).logits
        probs  = torch.nn.functional.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)
    return CLASS_NAMES[pred.item()], conf.item() * 100


# ──────────────────────────────────────────────────────────────
#  VOICE MAP — edge-tts Neural voices (Male / Female / Neutral)
# ──────────────────────────────────────────────────────────────
# Keys: language code → {"Female": voice_name, "Male": voice_name, "Neutral": voice_name}
# Voices sourced from Microsoft Edge Neural TTS catalogue.
_EDGE_VOICE_MAP = {
    "en": {
        "Female":  "en-US-JennyNeural",
        "Male":    "en-US-GuyNeural",
        "Neutral": "en-US-AriaNeural",
    },
    "hi": {
        "Female":  "hi-IN-SwaraNeural",
        "Male":    "hi-IN-MadhurNeural",
        "Neutral": "hi-IN-SwaraNeural",
    },
    "te": {
        "Female":  "te-IN-ShrutiNeural",
        "Male":    "te-IN-MohanNeural",
        "Neutral": "te-IN-ShrutiNeural",
    },
    "ta": {
        "Female":  "ta-IN-PallaviNeural",
        "Male":    "ta-IN-ValluvarNeural",
        "Neutral": "ta-IN-PallaviNeural",
    },
    "kn": {
        "Female":  "kn-IN-SapnaNeural",
        "Male":    "kn-IN-GaganNeural",
        "Neutral": "kn-IN-SapnaNeural",
    },
    "ml": {
        "Female":  "ml-IN-SobhanaNeural",
        "Male":    "ml-IN-MidhunNeural",
        "Neutral": "ml-IN-SobhanaNeural",
    },
    "bn": {
        "Female":  "bn-IN-TanishaaNeural",
        "Male":    "bn-IN-BashkarNeural",
        "Neutral": "bn-IN-TanishaaNeural",
    },
    "mr": {
        "Female":  "mr-IN-AarohiNeural",
        "Male":    "mr-IN-ManoharNeural",
        "Neutral": "mr-IN-AarohiNeural",
    },
    "gu": {
        "Female":  "gu-IN-DhwaniNeural",
        "Male":    "gu-IN-NiranjanNeural",
        "Neutral": "gu-IN-DhwaniNeural",
    },
    "ar": {
        "Female":  "ar-SA-ZariyahNeural",
        "Male":    "ar-SA-HamedNeural",
        "Neutral": "ar-SA-ZariyahNeural",
    },
    # Languages without edge-tts coverage fall back to gTTS below
    "pa": None,
    "as": None,
    "ur": None,
    "or": None,
    "sa": None,
}

# Module-level icon map used by all gender radio format_func lambdas
_GENDER_ICONS = {"Female": "👩", "Male": "👨", "Neutral": "🧑"}


def _get_voice(lang_code: str, gender: str) -> str | None:
    """Return the edge-tts voice name, or None if unsupported (use gTTS fallback)."""
    entry = _EDGE_VOICE_MAP.get(lang_code)
    if entry is None:
        return None
    return entry.get(gender, entry.get("Neutral"))


async def _edge_tts_async(text: str, voice: str) -> bytes:
    """Run edge-tts and return MP3 bytes."""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return buf.read()


# ──────────────────────────────────────────────────────────────
#  TTS HELPER  — edge-tts (neural) with gTTS fallback
# ──────────────────────────────────────────────────────────────
def tts_bytes(text: str, lang_code: str, gender: str = "Female") -> bytes:
    """Return MP3 bytes using edge-tts neural voice when available, else gTTS."""
    import asyncio
    voice = _get_voice(lang_code, gender)
    if voice:
        try:
            # Run async edge-tts
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, _edge_tts_async(text, voice))
                        return future.result(timeout=30)
                else:
                    return loop.run_until_complete(_edge_tts_async(text, voice))
            except RuntimeError:
                return asyncio.run(_edge_tts_async(text, voice))
        except Exception:
            pass  # Fall through to gTTS on any error
    # ── gTTS fallback ──────────────────────────────────────────
    from gtts import gTTS
    buf = io.BytesIO()
    gTTS(text=text, lang=lang_code, slow=False).write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# ──────────────────────────────────────────────────────────────
#  STT HELPER  — Whisper
# ──────────────────────────────────────────────────────────────
def stt_transcribe(audio_bytes: bytes, lang_code: str) -> str:
    whisper_model = load_whisper()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        result = whisper_model.transcribe(tmp, language=lang_code)
        return result.get("text", "").strip()
    finally:
        os.unlink(tmp)


# ──────────────────────────────────────────────────────────────
#  TRANSLATION HELPER
# ──────────────────────────────────────────────────────────────
def translate(text: str, target_code: str) -> str:
    try:
        return GoogleTranslator(source="auto", target=target_code).translate(text)
    except Exception as e:
        return f"[Translation error: {e}]"


# ══════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════
st.title("🤟 Sign Language Recognition and Communication System")
st.markdown("""
<div style='text-align:center;color:#667eea;font-size:1.05rem;margin-bottom:1.5rem'>
  Real-time ISL Detection &nbsp;•&nbsp; Multilingual Translation
  &nbsp;•&nbsp; Text-to-Speech &nbsp;•&nbsp; Speech-to-Text
</div>""", unsafe_allow_html=True)

_tabs_list = ["🎥 Live Detection", "🌐 Translation", "🔊 Text → Speech",
              "🎤 Speech → Text", "📚 ISL Dictionary"]
if st.session_state.get("user_role") == "admin":
    _tabs_list.append("🛡️ Admin Panel")

_all_tabs = st.tabs(_tabs_list)
tab_detect = _all_tabs[0]
tab_trans  = _all_tabs[1]
tab_tts    = _all_tabs[2]
tab_stt    = _all_tabs[3]
tab_dict   = _all_tabs[4]
tab_admin  = _all_tabs[5] if len(_all_tabs) > 5 else None

# ══════════════════════════════════════════════
#  TAB 1 – LIVE CAMERA DETECTION
# ══════════════════════════════════════════════
with tab_detect:
    st.header("🎥 Real-Time Sign Detection")

    # ── Detection mode toggle ──────────────────────────────────
    word_predictor = load_word_predictor()
    _mode_options = ["🔤 Letter Mode (A–Z, 1–9)"]
    if word_predictor:
        _mode_options.append("📝 Word Mode (Full Words)")
    else:
        _mode_options.append("📝 Word Mode (⚠️ Model not trained)")

    det_mode = st.radio(
        "Detection Mode", _mode_options, horizontal=True, key="det_mode"
    )
    is_word_mode = "Word Mode" in det_mode and word_predictor is not None

    if is_word_mode:
        st.markdown(
            "Recognise **whole ISL words** from gesture sequences. "
            "The LSTM model analyses a sequence of frames to identify words. "
            "Hold your sign steady for ~1 second."
        )
    else:
        st.markdown(
            "YOLOv8 draws a bounding box around each hand gesture. "
            "ViT then classifies the cropped sign. "
            "Detected signs automatically flow into the Translation and TTS pipeline below."
        )

    if "Word Mode" in det_mode and word_predictor is None:
        st.warning(
            "⚠️ **Word recognition model is not trained yet.** "
            "Run the training pipeline first:\n\n"
            "```bash\n"
            "pip install -r requirements.txt\n"
            "python -m word_recognition.download_data\n"
            "python -m word_recognition.extract_landmarks\n"
            "python -m word_recognition.train\n"
            "```"
        )

    col_feed, col_signs = st.columns(2)
    with col_feed:
        st.markdown('<div class="card"><h3>📹 Live Feed + Bounding Box</h3></div>',
                    unsafe_allow_html=True)
        frame_ph = st.empty()
    with col_signs:
        if is_word_mode:
            st.markdown('<div class="card-pink"><h3>📝 Recognized Words</h3></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="card-pink"><h3>✨ Recognized Signs</h3></div>',
                        unsafe_allow_html=True)
        signs_ph = st.empty()

    c1, c2, c3 = st.columns(3)
    with c1:
        start_btn = st.button("▶️ Start Detection", key="start_det",  use_container_width=True)
    with c2:
        stop_btn  = st.button("⏹️ Stop Detection",  key="stop_det",   use_container_width=True)
    with c3:
        clear_btn = st.button("🗑️ Clear Signs",      key="clear_signs", use_container_width=True)

    if start_btn: st.session_state["running"] = True
    if stop_btn:  st.session_state["running"] = False
    if clear_btn:
        st.session_state.pop("recognized_signs", None)
        st.session_state.pop("detected_word",    None)
        st.session_state.pop("word_predictions",  None)

    st.session_state.setdefault("running",          False)
    st.session_state.setdefault("recognized_signs", [])
    st.session_state.setdefault("word_predictions",  [])

    with st.expander("⚙️ Auto-Pipeline Settings"):
        pipeline_lang_name = st.selectbox(
            "Auto-translate signs to:", list(LANGUAGES.keys()), key="pipe_lang")
        auto_tts_on = st.checkbox(
            "🔊 Auto-generate TTS after detection", value=False, key="auto_tts")
        conf_thresh = st.slider(
            "Min confidence (%)", 30, 95, 60, key="conf_thr")

    # ── Live loop ──────────────────────────────────────────────
    if st.session_state["running"]:
        yolo = load_yolo()
        vit  = load_vit()
        cap  = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Cannot open webcam. Check camera permissions.")
            st.session_state["running"] = False
        else:
            pred_ph = st.empty()
            word_progress_ph = st.empty() if is_word_mode else None
            word_result_ph   = st.empty() if is_word_mode else None

            # Reset word predictor buffer on start
            if is_word_mode and word_predictor:
                word_predictor.reset_buffer()

            try:
                while st.session_state.get("running"):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)

                    annotated, crops = detect_and_crop(frame, yolo)
                    frame_ph.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        channels="RGB", use_column_width=True,
                    )

                    if is_word_mode and word_predictor:
                        # ── WORD MODE ──────────────────────────────
                        result = word_predictor.add_frame(frame)
                        if result:
                            progress = result["buffer_progress"]
                            word_progress_ph.progress(
                                progress,
                                text=f"Collecting frames… {int(progress * 100)}%"
                                     if progress < 1.0 else "Analysing gesture…"
                            )

                            if result["word"] and result["confidence"] >= conf_thresh:
                                word = result["word"]
                                conf = result["confidence"]

                                # Show prediction
                                word_result_ph.markdown(
                                    f'<div class="success-box">'
                                    f'📝 <b>{word}</b> ({conf:.1f}%)</div>',
                                    unsafe_allow_html=True,
                                )

                                # Add to word predictions list
                                preds = st.session_state["word_predictions"]
                                if not preds or preds[-1].split()[0] != word:
                                    preds.append(f"{word} ({conf:.1f}%)")
                                st.session_state["word_predictions"] = preds[-20:]
                                st.session_state["detected_word"] = " ".join(
                                    p.split()[0] for p in preds
                                )

                                # Show top-5
                                with signs_ph.container():
                                    st.markdown("**Latest prediction:**")
                                    for w, c in result.get("top_5", []):
                                        bar_pct = int(c)
                                        st.markdown(
                                            f"`{w}` — {c:.1f}%"
                                        )
                                    st.markdown("---")
                                    st.markdown("**Detected words:**")
                                    for p in st.session_state["word_predictions"][-10:]:
                                        st.write(f"• {p}")

                                # Reset buffer for next word
                                word_predictor.reset_buffer()

                    else:
                        # ── LETTER MODE ────────────────────────────
                        lines = []
                        for crop in crops:
                            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                            label, conf = classify_sign(pil, vit)
                            if conf >= conf_thresh:
                                lines.append(f"**{label}** ({conf:.1f}%)")
                                signs = st.session_state["recognized_signs"]
                                # Deduplicate consecutive same sign
                                if not signs or signs[-1].split()[0] != label:
                                    signs.append(f"{label} ({conf:.1f}%)")
                                st.session_state["recognized_signs"] = signs[-20:]
                                st.session_state["detected_word"] = " ".join(
                                    s.split()[0]
                                    for s in st.session_state["recognized_signs"]
                                )

                        pred_ph.markdown("\n\n".join(lines) or "*No sign detected*")
                        with signs_ph.container():
                            for s in st.session_state["recognized_signs"][-12:]:
                                st.write(f"• {s}")

                    time.sleep(0.04)
            finally:
                cap.release()
                st.session_state["running"] = False

    # ── Pipeline output ────────────────────────────────────────
    detected = st.session_state.get("detected_word", "")
    if detected:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("🔗 Auto-Pipeline Output")
        tgt_code   = LANGUAGES[pipeline_lang_name]
        translated = translate(detected, tgt_code)

        out_c1, out_c2 = st.columns(2)
        with out_c1:
            st.markdown(
                f'<div class="info-box">🔤 Detected signs: <b>{detected}</b></div>',
                unsafe_allow_html=True)
        with out_c2:
            st.markdown(
                f'<div class="success-box">🌐 {pipeline_lang_name}: <b>{translated}</b></div>',
                unsafe_allow_html=True)

        pipe_gender = st.radio(
            "🎙️ Voice gender for auto-speak:",
            ["Female", "Male", "Neutral"],
            format_func=lambda g: f"{_GENDER_ICONS[g]} {g}",
            horizontal=True,
            key="pipe_tts_gender",
        )
        if auto_tts_on or st.button("🔊 Speak detected signs", key="speak_det"):
            with st.spinner(f"Generating {pipe_gender.lower()} voice…"):
                audio = tts_bytes(translated, tgt_code, gender=pipe_gender)
            st.audio(audio, format="audio/mp3")


# ══════════════════════════════════════════════
#  TAB 2 – TRANSLATION
# ══════════════════════════════════════════════
with tab_trans:
    st.header("🌐 Text Translation")

    prefill = st.session_state.get("detected_word", "")

    t_c1, t_c2 = st.columns(2)
    with t_c1:
        src_text = st.text_area("📝 Source text (any language)",
                                value=prefill, height=150, key="trans_src")
    with t_c2:
        tgt_lang_name = st.selectbox("🗣️ Translate to", list(LANGUAGES.keys()), key="trans_tgt")

    trans_gender = st.radio(
        "🎙️ Voice gender for spoken translation:",
        ["Female", "Male", "Neutral"],
        format_func=lambda g: f"{_GENDER_ICONS[g]} {g}",
        horizontal=True,
        key="trans_gender",
    )

    if st.button("✅ Translate", key="trans_btn"):
        if src_text.strip():
            code = LANGUAGES[tgt_lang_name]
            with st.spinner("Translating…"):
                result = translate(src_text.strip(), code)
            st.markdown(
                f'<div class="info-box">📤 <b>{tgt_lang_name}:</b> {result}</div>',
                unsafe_allow_html=True)
            st.session_state["last_translation"] = result
            st.session_state["last_trans_lang"]   = tgt_lang_name
            # Speak the translation with chosen gender voice
            with st.spinner(f"Generating {trans_gender.lower()} voice…"):
                audio = tts_bytes(result, code, gender=trans_gender)
            st.audio(audio, format="audio/mp3")
            st.markdown('<div class="success-box">✅ Translation complete with audio!</div>',
                        unsafe_allow_html=True)
        else:
            st.warning("Enter some text first.")


# ══════════════════════════════════════════════
#  TAB 3 – TEXT → SPEECH
# ══════════════════════════════════════════════
with tab_tts:
    st.header("🔊 Text → Speech")
    st.markdown(
        "Uses **Microsoft Edge Neural TTS** for natural-sounding voices when available, "
        "with **gTTS** as fallback. Select your preferred voice gender below."
    )

    # ── Voice gender selector ──────────────────────────────────
    _GENDER_ICONS = {"Female": "👩", "Male": "👨", "Neutral": "🧑"}
    tts_gender = st.radio(
        "🎙️ Voice Gender",
        options=["Female", "Male", "Neutral"],
        format_func=lambda g: f"{_GENDER_ICONS[g]} {g}",
        horizontal=True,
        key="tts_gender",
    )

    # Show voice name being used
    _preview_lang = st.session_state.get("tts_lang_sel", "English")
    _preview_code = LANGUAGES.get(_preview_lang, "en")
    _preview_voice = _get_voice(_preview_code, tts_gender)
    if _preview_voice:
        st.caption(f"🎤 Voice: `{_preview_voice}` (Microsoft Edge Neural TTS)")
    else:
        st.caption(f"🎤 Voice: Google TTS fallback (neural voices not available for this language)")

    st.markdown("<hr style='margin:0.6rem 0'>", unsafe_allow_html=True)

    prefill_tts = st.session_state.get(
        "last_translation",
        st.session_state.get("detected_word", "")
    )

    tts_c1, tts_c2 = st.columns([2, 1])
    with tts_c1:
        tts_input = st.text_area("📝 Text to speak", value=prefill_tts,
                                  height=150, key="tts_input")
    with tts_c2:
        tts_lang_sel = st.selectbox("🗣️ Output language",
                                     list(LANGUAGES.keys()), key="tts_lang_sel")
        st.markdown("<br>", unsafe_allow_html=True)
        tts_go = st.button("🔊 Generate Speech", key="tts_go", use_container_width=True)

    if tts_go:
        if tts_input.strip():
            code = LANGUAGES[tts_lang_sel]
            with st.spinner("Preparing text…"):
                translated_for_tts = (
                    translate(tts_input.strip(), code)
                    if code != "en" else tts_input.strip()
                )
            st.markdown(
                f'<div class="info-box">🎯 Speaking in {tts_lang_sel} ({tts_gender} voice):<br>{translated_for_tts}</div>',
                unsafe_allow_html=True)
            with st.spinner(f"Synthesising {tts_gender.lower()} voice…"):
                audio = tts_bytes(translated_for_tts, code, gender=tts_gender)
            st.audio(audio, format="audio/mp3")
            st.markdown('<div class="success-box">✅ Audio ready!</div>',
                        unsafe_allow_html=True)
        else:
            st.warning("Enter some text first.")

    # Quick shortcut from camera
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("⚡ Quick Speak — Camera Output")
    dw = st.session_state.get("detected_word", "")
    if dw:
        st.info(f"Camera detected: **{dw}**")
        qs_c1, qs_c2 = st.columns(2)
        with qs_c1:
            q_lang = st.selectbox("Speak in:", list(LANGUAGES.keys()), key="quick_tts_lang")
        with qs_c2:
            q_gender = st.radio(
                "Voice:",
                ["Female", "Male", "Neutral"],
                format_func=lambda g: f"{_GENDER_ICONS[g]} {g}",
                horizontal=True,
                key="quick_tts_gender",
            )
        if st.button("🔊 Speak", key="quick_tts"):
            code  = LANGUAGES[q_lang]
            txt   = translate(dw, code)
            audio = tts_bytes(txt, code, gender=q_gender)
            st.audio(audio, format="audio/mp3")
    else:
        st.info("No camera detections yet — start Live Detection to feed this panel.")


# ══════════════════════════════════════════════
#  TAB 4 – SPEECH → TEXT  (Whisper)
# ══════════════════════════════════════════════
with tab_stt:
    st.header("🎤 Speech → Text  (OpenAI Whisper)")
    st.markdown("""
    Whisper supports **90+ languages** including every major Indian language.  
    Upload a WAV/MP3 file, or record live with your microphone.  
    The transcript can then be translated and spoken aloud.
    """)

    stt_method = st.radio(
        "Input method:",
        ["📁 Upload audio file", "🎤 Record from microphone"],
        horizontal=True, key="stt_method"
    )

    stt_lang_sel  = st.selectbox("🗣️ Spoken language in audio",
                                  list(WHISPER_LANGS.keys()), key="stt_lang")
    stt_lang_code = WHISPER_LANGS[stt_lang_sel]

    audio_for_stt = None

    if stt_method == "📁 Upload audio file":
        uploaded = st.file_uploader(
            "Upload audio (WAV, MP3, M4A, OGG, FLAC)",
            type=["wav","mp3","m4a","ogg","flac"], key="stt_upload"
        )
        if uploaded:
            audio_for_stt = uploaded.read()
            st.audio(audio_for_stt)

    else:
        st.markdown("""
        **Live microphone** requires the `audio-recorder-streamlit` package.  
        Install it once with:
        ```
        pip install audio-recorder-streamlit
        ```
        """)
        try:
            from audio_recorder_streamlit import audio_recorder
            recorded = audio_recorder(
                text="🎤 Click to record",
                recording_color="#667eea",
                neutral_color="#764ba2",
                icon_size="2x",
            )
            if recorded:
                audio_for_stt = recorded
                st.audio(audio_for_stt, format="audio/wav")
        except ImportError:
            st.warning("⚠️ `audio-recorder-streamlit` not installed. "
                       "Upload an audio file instead, or run: "
                       "`pip install audio-recorder-streamlit`")

    if audio_for_stt:
        if st.button("📝 Transcribe with Whisper", key="stt_go", use_container_width=True):
            with st.spinner("Transcribing… (this may take a moment)"):
                transcript = stt_transcribe(audio_for_stt, stt_lang_code)

            st.markdown(
                f'<div class="success-box">📝 Transcript:<br>{transcript}</div>',
                unsafe_allow_html=True)
            st.session_state["stt_transcript"] = transcript

    # Post-STT pipeline
    transcript = st.session_state.get("stt_transcript", "")
    if transcript:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("🔗 Translate + Speak Transcript")
        p_c1, p_c2 = st.columns([1, 1])
        with p_c1:
            post_lang_sel = st.selectbox("Translate to:",
                                          list(LANGUAGES.keys()), key="post_stt_lang")
        with p_c2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🌐 Translate & 🔊 Speak", key="post_stt_go", use_container_width=True):
                code = LANGUAGES[post_lang_sel]
                with st.spinner("Translating…"):
                    translated = translate(transcript, code)
                st.markdown(
                    f'<div class="info-box">🌐 {post_lang_sel}:<br>{translated}</div>',
                    unsafe_allow_html=True)
                with st.spinner("Generating speech…"):
                    audio = tts_bytes(translated, code)
                st.audio(audio, format="audio/mp3")


# ══════════════════════════════════════════════
#  TAB 5 – ISL DICTIONARY
# ══════════════════════════════════════════════
with tab_dict:
    st.header("📚 ISL Dictionary")
    st.markdown("""
    Browse ISL video examples organised by letter (A–Z) and Numbers.  
    Data source: `ISL_Dictionary_words.csv`
    """)

    # ── Word model status banner ──────────────────────────────
    _wp = load_word_predictor()
    if _wp and _wp.is_loaded:
        st.markdown(
            f'<div class="success-box">'
            f'✅ Word recognition model loaded — <b>{_wp.num_classes}</b> words available'
            f'</div>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="info-box">'
            '⚠️ Word model not trained yet — only letter/number recognition available. '
            'Run the training pipeline to enable word recognition.'
            '</div>', unsafe_allow_html=True
        )

    # ── Tabs: Letters vs Words ────────────────────────────────
    dict_tab_letters, dict_tab_words = st.tabs(["🔤 Letters & Numbers", "📝 Word Vocabulary"])

    with dict_tab_letters:
        query = st.text_input("🔍 Search letter / category", "", key="dict_letter_search").strip().upper()
        items = [
            (name, link) for name, link in letter_links.items()
            if query in name.upper() or not query
        ]

        if not items:
            st.warning("No results. Try a different letter.")
        else:
            cols_n = 4
            for i in range(0, len(items), cols_n):
                chunk = items[i : i + cols_n]
                cols  = st.columns(cols_n)
                for col, (name, link) in zip(cols, chunk):
                    with col:
                        st.markdown(f"""
                        <div style='background:linear-gradient(135deg,#667eea,#764ba2);
                                    padding:1.3rem;border-radius:12px;
                                    text-align:center;margin-bottom:.8rem'>
                          <h3 style='color:white;margin:0;font-size:2rem'>{name}</h3>
                          <a href='{link}' target='_blank'
                             style='color:rgba(255,255,255,.85);font-size:.85rem;
                                    text-decoration:none'>📂 Open Drive Folder</a>
                        </div>""", unsafe_allow_html=True)

    with dict_tab_words:
        if _wp and _wp.is_loaded:
            all_words = _wp.available_words
            st.markdown(f"**{len(all_words)}** words available for recognition.")

            # Search
            word_query = st.text_input(
                "🔍 Search words", "", key="dict_word_search"
            ).strip().lower()

            filtered = [w for w in all_words if word_query in w.lower()] if word_query else all_words

            if not filtered:
                st.warning("No matching words found.")
            else:
                # Group by first letter
                from collections import defaultdict
                by_letter = defaultdict(list)
                for w in filtered:
                    first = w[0].upper() if w else "?"
                    by_letter[first].append(w)

                for letter in sorted(by_letter.keys()):
                    words = by_letter[letter]
                    with st.expander(f"**{letter}** — {len(words)} words", expanded=bool(word_query)):
                        # Display as a wrapped grid of tags
                        tags_html = " ".join(
                            f"<span style='display:inline-block;background:linear-gradient(135deg,#667eea,#764ba2);"
                            f"color:white;padding:0.3rem 0.8rem;border-radius:20px;"
                            f"margin:0.2rem;font-size:0.85rem;font-weight:500'>{w}</span>"
                            for w in words
                        )
                        st.markdown(tags_html, unsafe_allow_html=True)
        else:
            st.info(
                "📋 **Word vocabulary will appear here after training.**\n\n"
                "Run the training pipeline to build the word dictionary:\n\n"
                "```bash\n"
                "python -m word_recognition.download_data\n"
                "python -m word_recognition.extract_landmarks\n"
                "python -m word_recognition.train\n"
                "```"
            )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("📋 Full Table")
    display_df = isl_df.copy()
    display_df.columns = [c.strip() for c in display_df.columns]
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
#  ADMIN PANEL  (admin role only)
# ══════════════════════════════════════════════
if tab_admin is not None:
    with tab_admin:
        st.header("🛡️ Admin Panel — User Management")
        st.markdown(
            "Manage all registered users. You can view details, approve/suspend accounts, "
            "promote to admin, or delete users."
        )

        # ── Refresh button ──────────────────────────────────────
        if st.button("🔄 Refresh User List", key="admin_refresh"):
            st.rerun()

        with _get_conn() as _conn:
            _rows = _conn.execute(
                "SELECT id, username, display_name, email, role, is_approved, created_at, last_login "
                "FROM users ORDER BY created_at DESC"
            ).fetchall()

        all_users = [dict(r) for r in _rows]

        # ── Summary metrics ─────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("👥 Total Users",    len(all_users))
        m2.metric("✅ Active",          sum(1 for u in all_users if u["is_approved"]))
        m3.metric("⏳ Pending",         sum(1 for u in all_users if not u["is_approved"]))
        m4.metric("🛡️ Admins",          sum(1 for u in all_users if u["role"] == "admin"))

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── User table ──────────────────────────────────────────
        st.subheader("📋 Registered Users")
        for user in all_users:
            is_self = user["username"] == st.session_state["username"]
            with st.expander(
                f"{'🛡️' if user['role']=='admin' else '👤'} "
                f"{user['display_name']}  (@{user['username']})"
                f"{'  ← YOU' if is_self else ''}"
                f"  |  {'✅ Active' if user['is_approved'] else '⏳ Pending'}",
                expanded=False,
            ):
                i1, i2, i3 = st.columns(3)
                i1.markdown(f"**Email:** {user['email'] or '—'}")
                i2.markdown(f"**Role:** `{user['role']}`")
                i3.markdown(f"**Joined:** {user['created_at'][:10]}")
                st.markdown(f"**Last login:** {user['last_login'] or 'Never'}")

                if not is_self:
                    ac1, ac2, ac3 = st.columns(3)

                    # Approve / Suspend
                    if user["is_approved"]:
                        if ac1.button("🚫 Suspend", key=f"susp_{user['id']}"):
                            with _get_conn() as c:
                                c.execute("UPDATE users SET is_approved=0 WHERE id=?",
                                          (user["id"],))
                                c.commit()
                            st.success(f"Suspended @{user['username']}.")
                            st.rerun()
                    else:
                        if ac1.button("✅ Approve", key=f"appr_{user['id']}"):
                            with _get_conn() as c:
                                c.execute("UPDATE users SET is_approved=1 WHERE id=?",
                                          (user["id"],))
                                c.commit()
                            st.success(f"Approved @{user['username']}.")
                            st.rerun()

                    # Toggle role
                    new_role = "user" if user["role"] == "admin" else "admin"
                    role_label = "⬇️ Demote to User" if user["role"] == "admin" else "⬆️ Make Admin"
                    if ac2.button(role_label, key=f"role_{user['id']}"):
                        with _get_conn() as c:
                            c.execute("UPDATE users SET role=? WHERE id=?",
                                      (new_role, user["id"]))
                            c.commit()
                        st.success(f"Role updated for @{user['username']} → {new_role}.")
                        st.rerun()

                    # Delete
                    if ac3.button("🗑️ Delete", key=f"del_{user['id']}",
                                   type="primary"):
                        with _get_conn() as c:
                            c.execute("DELETE FROM users WHERE id=?", (user["id"],))
                            c.commit()
                        st.success(f"Deleted @{user['username']}.")
                        st.rerun()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Add user manually ───────────────────────────────────
        st.subheader("➕ Add User Manually")
        with st.form("admin_add_user"):
            na_c1, na_c2 = st.columns(2)
            na_user    = na_c1.text_input("Username *",     key="na_user")
            na_display = na_c2.text_input("Display Name",    key="na_display")
            na_email   = na_c1.text_input("Email",           key="na_email")
            na_role    = na_c2.selectbox("Role", ["user", "admin"], key="na_role")
            na_pw      = na_c1.text_input("Password *", type="password", key="na_pw")
            na_submit  = st.form_submit_button("➕ Add User", use_container_width=True)

        if na_submit:
            if not na_user or not na_pw:
                st.error("Username and password are required.")
            else:
                try:
                    with _get_conn() as c:
                        c.execute("""
                            INSERT INTO users
                              (username, password_hash, display_name, email, role, is_approved, created_at)
                            VALUES (?, ?, ?, ?, ?, 1, ?)
                        """, (na_user.strip().lower(), _hash(na_pw),
                              na_display.strip() or na_user,
                              na_email.strip(), na_role,
                              datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                        c.commit()
                    st.success(f"✅ User @{na_user} added successfully.")
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error("❌ Username already exists.")
