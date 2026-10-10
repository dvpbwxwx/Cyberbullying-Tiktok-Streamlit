import os
import pandas as pd
import re
import string
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. KONFIGURASI HALAMAN STREAMLIT
st.set_page_config(
    page_title="Analisis Sentimen TikTok - IndoBERT",
    page_icon="🤖",
    layout="wide"
)

# 2. CSS & BOOTSTRAP
bootstrap_and_custom_css = """
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    body, .stApp {
        font-family: 'Poppins', sans-serif !important;
        background-color: #f8f9fa;
    }
    
    .hero-banner {
        background: linear-gradient(135deg, #405590 0%, #1e2640 100%);
        color: white;
        padding: 60px 40px;
        border-radius: 20px;
        margin-bottom: 40px;
        box-shadow: 0 10px 30px rgba(64, 85, 144, 0.2);
    }
    
    .stat-card {
        transition: all 0.3s ease;
        border-bottom: 5px solid #405590;
        background: white;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        text-align: center;
    }
    .stat-card:hover {
        transform: translateY(-8px);
    }
    
    .result-card-custom {
        border: none;
        border-radius: 20px;
        background: linear-gradient(145deg, #ffffff, #f0f4f8);
        box-shadow: 0 8px 25px rgba(0,0,0,0.05);
        padding: 30px;
        margin-top: 25px;
    }
    
    .sentiment-badge-bullying {
        font-size: 2.2rem;
        font-weight: 700;
        color: #dc3545;
    }
    
    .sentiment-badge-safe {
        font-size: 2.2rem;
        font-weight: 700;
        color: #198754;
    }
    
    .confidence-text {
        color: #405590;
        font-weight: 600;
        font-size: 1.5rem;
    }
</style>
"""
st.markdown(bootstrap_and_custom_css, unsafe_allow_html=True)

# 3. HERO BANNER
st.markdown("""
<div class="hero-banner text-center">
    <h3 style="letter-spacing: 2px; font-weight: 300; font-size: 20px;">ANALISIS SENTIMEN</h3>
    <h1 style="font-size: 40px; font-weight: 800; margin: 15px 0;">Cyberbullying Pada TikTok Menggunakan IndoBERT</h1>
    <p style="font-size: 16px; opacity: 0.9; max-width: 800px; margin: 0 auto;">
        Menggunakan arsitektur Transformer (IndoBERT) untuk pemahaman bahasa yang lebih kontekstual dalam mendeteksi ujaran kebencian secara otomatis.
    </p>
</div>
""", unsafe_allow_html=True)

# 4. LOAD MODEL INDOBERT
MODEL_PATH = "dappa2999/indobert-tiktok-cyberbullying"

@st.cache_resource
def load_model_and_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        return tokenizer, model
    except Exception as e:
        MODEL_NAME = "indobenchmark/indobert-base-p1"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()
        return tokenizer, model

with st.spinner("Deep Learning sedang memproses model IndoBERT..."):
    tokenizer, model = load_model_and_tokenizer()

# 5. STATISTIC CARD
st.markdown("""
<div class="row mb-5">
    <div class="col-lg-4 mb-3">
        <div class="stat-card">
            <div class="mb-2"><i class="fa fa-database text-info fs-3"></i></div>
            <h2 class="fw-bold" style="color: #1e2640;">8.198</h2>
            <p class="text-dark mb-0 fw-semibold">Total Dataset</p>
            <small class="text-muted">Setelah Random Over-Sampling (ROS)</small>
        </div>
    </div>
    <div class="col-lg-4 mb-3">
        <div class="stat-card">
            <div class="mb-2"><i class="fa fa-rocket text-danger fs-3"></i></div>
            <h2 class="fw-bold" style="color: #1e2640;">91.10%</h2>
            <p class="text-dark mb-0 fw-semibold">Akurasi IndoBERT</p>
            <small class="text-muted">Skenario Zero-Shot Classification</small>
        </div>
    </div>
    <div class="col-lg-4 mb-3">
        <div class="stat-card">
            <div class="mb-2"><i class="fa fa-microchip text-success fs-3"></i></div>
            <h2 class="fw-bold" style="color: #1e2640;">80:10:10</h2>
            <p class="text-dark mb-0 fw-semibold">Data Split</p>
            <small class="text-muted">Train, Validation, Test</small>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# METODE & EVALUATION
st.markdown("""
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<style>
    /* Reset total margin/padding accordion */
    .custom-accordion .accordion-item {
        background: white !important;
        border-radius: 15px !important;
        overflow: hidden;
        margin-bottom: 15px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .custom-accordion .accordion-button {
        background: white !important;
        color: #405590 !important;
        font-weight: 700 !important;
        padding: 15px 20px 5px 20px !important; /* Padding bawah dibuat sangat kecil */
        margin: 0 !important;
        box-shadow: none !important;
    }
    .custom-accordion .accordion-button:after {
        margin-top: -10px;
    }
    .custom-accordion .accordion-body {
        padding: 2px 20px 15px 20px !important; /* Padding atas rapat dengan judul */
        margin: 0 !important;
    }
    /* Sembunyikan ikon anchor bawaan Streamlit */
    .stApp a.header-anchor {
        display: none !important;
    }
</style>
<div class="container mb-5 custom-accordion" style="margin-top: -10px;">
    <div class="row align-items-start mb-5">
        <div class="col-lg-6 mb-4">
            <div class="section-heading text-start mb-3">
                <h3 class='text-left fw-bold' style='color: #405590;'>Metode Penelitian</h3>
                <p class="text-dark small m-0">Penelitian ini membandingkan metode pelabelan <strong>Zero-Shot Classification</strong> dan <strong>InSet Lexicon</strong>, di mana hasil terbaik dicapai oleh IndoBERT pada skenario Zero-Shot dengan teknik penyeimbangan data (ROS).</p>
            </div>
            <div class="accordion-item">
                <div class="accordion-header">
                    <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#collapseOne">
                        <i class="fa fa-brain me-2" style="color: #405590;"></i> Keunggulan IndoBERT
                    </button>
                </div>
                <div id="collapseOne" class="accordion-collapse collapse show">
                    <div class="accordion-body small text-muted">
                        Arsitektur <i>Transformer</i> pada IndoBERT menggunakan mekanisme <i>self-attention</i> yang sangat efektif dalam mengenali konteks ujaran kebencian di TikTok, terutama setelah di-fine-tuning dengan data yang seimbang.
                    </div>
                </div>
            </div>
            <div class="accordion-item">
                <div class="accordion-header">
                    <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#distribusiCollapse">
                        <i class="fa fa-chart-pie me-2" style="color: #405590;"></i> Distribusi Dataset
                    </button>
                </div>
                <div id="distribusiCollapse" class="accordion-collapse collapse show">
                    <div class="accordion-body">
                        <div class="mb-3">
                            <div class="d-flex justify-content-between small mb-1">
                                <span class="fw-bold">Non-Bullying</span>
                                <span>4.099 Komentar (69.9%)</span>
                            </div>
                            <div class="progress" style="height: 10px;">
                                <div class="progress-bar bg-success" style="width: 69.9%"></div>
                            </div>
                        </div>
                        <div>
                            <div class="d-flex justify-content-between small mb-1">
                                <span class="fw-bold">Bullying</span>
                                <span>1.765 Komentar (30.1%)</span>
                            </div>
                            <div class="progress" style="height: 10px;">
                                <div class="progress-bar bg-danger" style="width: 30.1%"></div>
                            </div>
                        </div>
                        <p class="x-small text-muted mt-3 mb-0" style="font-size: 11px;">Total Dataset: 5.864 (Skenario ZSC)</p>
                    </div>
                </div>
            </div>
        </div>
        <div class="col-lg-6 mb-4">
            <div class="section-heading text-start mb-3">
                <h3 class='text-left fw-bold' style='color: #405590;'>Evaluasi Model</h3>
                <p class="text-dark small m-0">Hasil pengujian akhir pada data uji (test set) menunjukkan performa yang sangat stabil pada kedua kelas.</p>
            </div>
            <div class="accordion-item">
                <div class="accordion-header">
                    <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#matrixCollapse">
                        <i class="fa fa-th me-2" style="color: #405590;"></i> Confusion Matrix (IndoBERT)
                    </button>
                </div>
                <div id="matrixCollapse" class="accordion-collapse collapse show">
                    <div class="accordion-body">
                        <table class="table table-bordered text-center x-small mb-0" style="font-size: 12px;">
                            <thead class="table-light">
                                <tr>
                                    <th class="small">Aktual \ Prediksi</th>
                                    <th class="text-danger small">Bullying</th>
                                    <th class="text-success small">Non-Bullying</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td class="fw-bold small">Bullying</td>
                                    <td class="fw-bold" style="background-color: #ffebee !important;">372</td>
                                    <td>38</td>
                                </tr>
                                <tr>
                                    <td class="fw-bold small">Non-Bullying</td>
                                    <td>35</td>
                                    <td class="fw-bold" style="background-color: #e8f5e9 !important;">375</td>
                                </tr>
                            </tbody>
                        </table>
                        <p class="x-small text-muted mt-2 mb-0" style="font-size: 11px;">*Berdasarkan Support: 820 data uji</p>
                    </div>
                </div>
            </div>
            <div class="accordion-item">
                <div class="accordion-header">
                    <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#performaCollapse">
                        <i class="fa fa-check-double me-2" style="color: #405590;"></i> Detail Performa (IndoBERT ZSC)
                    </button>
                </div>
                <div id="performaCollapse" class="accordion-collapse collapse show">
                    <div class="accordion-body p-0">
                        <table class="table table-hover mb-0 x-small" style="font-size: 12px;">
                            <thead class="table-light">
                                <tr>
                                    <th>Metrik</th>
                                    <th class="text-center">Hasil</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr><td>Accuracy</td><td class="text-center fw-bold text-info">91.10%</td></tr>
                                <tr><td>Precision (Avg)</td><td class="text-center fw-bold">91.10%</td></tr>
                                <tr><td>Recall (Avg)</td><td class="text-center fw-bold">91.10%</td></tr>
                                <tr><td>F1-Score (Avg)</td><td class="text-center fw-bold">91.10%</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# SYSTEM WORKFLOW
st.markdown("<h2 class='text-center fw-bold' style='color: #405590;'>Alur Kerja Sistem</h2>", unsafe_allow_html=True)
st.markdown("<p class='text-center text-muted mb-4'>Tahapan pemrosesan teks menggunakan Pre-Trained Transformer & Zero-Shot Classification</p>", unsafe_allow_html=True)
st.markdown("""
<div class="container mb-5">
    <div class="row text-center">
        <div class="col-lg-3 col-md-6 mb-3">
            <div class="stat-card p-4 shadow-sm rounded-20 bg-white h-100">
                <div class="icon-stat mb-2"><i class="fa fa-broom text-info" style="color: #405590 !important;"></i></div>
                <h6 class="fw-bold">Preprocessing</h6>
                <p class="text-dark x-small mb-0">Pembersihan PII (username/@), noise, case folding, dan normalisasi slang TikTok tanpa menghilangkan konteks kalimat.</p>
            </div>
        </div>
        <div class="col-lg-3 col-md-6 mb-3">
            <div class="stat-card p-4 shadow-sm rounded-20 bg-white h-100">
                <div class="icon-stat mb-2"><i class="fa fa-tags text-info mb-2" style="color: #405590 !important;"></i></div>
                <h6 class="fw-bold">Zero-Shot Labeling</h6>
                <p class="text-dark x-small mb-0">Pelabelan kontekstual otomatis berbasis mDeBERTa-v3 yang tervalidasi Cohen's Kappa (K = 0.9132) hampir sempurna.</p>
            </div>
        </div>
        <div class="col-lg-3 col-md-6 mb-3">
            <div class="stat-card p-4 shadow-sm rounded-20 bg-white h-100">
                <div class="icon-stat mb-2"><i class="fa fa-brain text-info mb-2" style="color: #405590 !important;"></i></div>
                <h6 class="fw-bold">IndoBERT Model</h6>
                <p class="text-dark x-small mb-0">Ekstraksi fitur semantik menggunakan model IndoBERT (Indo4B) yang responsif terhadap konteks bahasa gaul media sosial.</p>
            </div>
        </div>
        <div class="col-lg-3 col-md-6 mb-3">
            <div class="stat-card p-4 shadow-sm rounded-20 bg-white h-100">
                <div class="icon-stat mb-2"><i class="fa fa-shield-alt text-info mb-2" style="color: #405590 !important;"></i></div>
                <h6 class="fw-bold">Hasil Klasifikasi</h6>
                <p class="text-dark x-small mb-0">Penentuan label akhir kategori Bullying (1) atau Non-Bullying (0) dengan tingkat akurasi model mencapai 91,10%.</p>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 6. DICTIONARY ALAY & SLANG

kamus_alay = {
    "ga": "tidak", "gak": "tidak", "ngga": "tidak", "nggak": "tidak", "tdk": "tidak",
    "gk": "tidak", "g": "tidak", "tp": "tapi", "tpi": "tapi", "klo": "kalau",
    "kalo": "kalau", "yg": "yang", "jd": "jadi", "jdi": "jadi", "krn": "karena",
    "karna": "karena", "dgn": "dengan", "dg": "dengan", "bkn": "bukan",
    "bgt": "banget", "bngt": "banget", "bgtt": "banget", "bagus": "bagus",
    "bgus": "bagus", "jlk": "jelek", "jele": "jelek", "ancur": "hancur",
    "kren": "keren", "bnr": "benar", "bener": "benar", "sm": "sama",
    "bnyk": "banyak", "sdh": "sudah", "udah": "sudah", "udh": "sudah",
    "blm": "belum", "belom": "belum", "gpp": "tidak apa apa",
    "sy": "saya", "gw": "saya", "gue": "saya", "gua": "saya", "aku": "saya",
    "lu": "kamu", "lo": "kamu", "loe": "kamu", "km": "kamu", "u": "kamu",
    "dr": "dari", "dri": "dari", "pd": "pada", "dlm": "dalam", "ngab": "bang",
    "cegil": "perempuan obsesif", "cogil": "laki laki obsesif", "fomo": "ikut ikutan",
    "redflag": "berbahaya", "red flag": "berbahaya", "greenflag": "baik",
    "ytjt": "yang tahu tahu saja", "fyp": "rekomendasi", "anjay": "keren",
    "anjir": "parah", "pls": "tolong", "jujurly": "sejujurnya",
    "sengaruh": "berpengaruh", "caper": "cari perhatian", "baper": "bawa perasaan",
    "gaje": "tidak jelas", "mager": "malas", "pargoy": "joget",
    "spill": "bocorkan", "salty": "kesal", "toxic": "beracun",
    "cringe": "aneh", "vibes": "suasana", "worth it": "pantas"
}

def clean_text(text):
    if not isinstance(text, str) or text is None:
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def normalisasi_alay(text):
    words = text.split()
    normalized_words = [kamus_alay.get(w, w) for w in words]
    return " ".join(normalized_words)

def predict_sentiment(text):
    cleaned = clean_text(text)
    final_text = normalisasi_alay(cleaned)
    
    inputs = tokenizer(final_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)
    
    sentiment_labels = ["non-bullying", "bullying"]
    return sentiment_labels[predicted_class.item()], confidence.item()

# 7. SENTIMENT ANALYSIS
st.markdown("""
<style>
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border-radius: 25px !important;
        border: none !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.06) !important;
        padding: 40px 50px !important;
        max-width: 850px !important;
        margin: 0 auto !important;
    }

    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        display: flex !important;
        width: 100% !important;
        background-color: #f8f9fa !important;
        padding: 6px !important;
        border-radius: 12px !important;
        border: none !important;
        margin-bottom: 30px !important;
        gap: 0px !important;
    }

    .stTabs [data-baseweb="tab-list"] button,
    .stTabs [data-baseweb="tab"] {
        flex: 1 1 50% !important;
        width: 50% !important;
        max-width: 50% !important;
        height: 48px !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        color: #666666 !important;
        background-color: transparent !important;
        border: none !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        transition: all 0.3s ease !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #405590 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(64, 85, 144, 0.25) !important;
    }

    .stButton > button {
        background-color: #405590 !important;
        color: white !important;
        border-radius: 30px !important;
        padding: 12px 40px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(64, 85, 144, 0.3) !important;
    }
    
    .stButton > button:hover {
        background-color: #314270 !important;
    }

    .stTextArea textarea {
        border-radius: 15px !important;
        border: 1px solid #e2e8f0 !important;
        padding: 15px !important;
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div id='analisis'></div>", unsafe_allow_html=True)
st.markdown("<h2 class='text-center fw-bold' style='color: #405590;'>Metode Analisis</h2>", unsafe_allow_html=True)
st.markdown("<p class='text-center text-muted mb-4'>Pilih analisis teks tunggal atau proses batch file CSV</p>", unsafe_allow_html=True)

with st.container(border=True):

    tab1, tab2 = st.tabs(["Analisis Teks", "Analisis Batch (CSV)"])

    # TAB 1: ANALYSIS TEXT
    with tab1:
        st.markdown("<p class='fw-bold text-start mb-2' style='color: #1a202c; font-size: 14px;'>Masukkan komentar TikTok untuk dianalisis:</p>", unsafe_allow_html=True)
        raw_text = st.text_area(
            label="", 
            placeholder="Tulis komentar di sini...", 
            height=120, 
            label_visibility="collapsed"
        )
        
        col_b1, col_b2, col_b3 = st.columns([1, 1.8, 1])
        with col_b2:
            btn_manual = st.button("🔍 Analisis Sekarang", key="btn_manual", use_container_width=True)
            
        if btn_manual:
            if raw_text.strip():
                with st.spinner("Deep Learning sedang memproses..."):
                    label, score = predict_sentiment(raw_text)
                    confidence_pct = round(score * 100, 2)
                    badge_class = "text-danger" if label == "bullying" else "text-success"
                
                # Predict Results
                st.markdown(f"""
                <div class="result-card-custom text-start mt-4 p-4" style="background: #f8fafd; border-radius: 20px; border: 1px solid #e2e8f0;">
                    <div class="row align-items-center">
                        <div class="col-md-6">
                            <span class="text-muted small">Hasil Analisis Sentimen:</span>
                            <div class="{badge_class}" style="font-size: 2.2rem; font-weight: 700;">{label.upper()}</div>
                            <p class="text-muted italic mb-0">"{raw_text}"</p>
                        </div>
                        <div class="col-md-6 text-md-end text-center mt-3 mt-md-0">
                            <span class="text-muted small">Confidence Score:</span>
                            <div style="color: #405590; font-weight: 600; font-size: 1.5rem;">{confidence_pct}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(score)
            else:
                st.warning("Silakan masukkan teks komentar terlebih dahulu.")

    # TAB 2: ANALYSIS BATCH CSV 
    with tab2:
        st.markdown("""
        <div class="alert alert-info border-0 shadow-sm rounded-15 text-start mb-4" style="border-radius: 15px; background-color: #e0f2fe; color: #0369a1;">
            <div class="d-flex align-items-center">
                <div class="me-3 fs-3">ℹ️</div>
                <div>
                    <h6 class="fw-bold mb-1">Panduan Unggah File:</h6>
                    <ul class="small mb-0 ps-3">
                        <li>Pastikan file berformat <strong>.csv</strong>.</li>
                        <li>Gunakan kolom pertama untuk teks komentar atau beri nama kolom <strong>"text_komentar"</strong>.</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload file CSV:", type=["csv"])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            col_name = 'text_komentar' if 'text_komentar' in df.columns else df.columns[0]
            
            col_b1, col_b2, col_b3 = st.columns([1, 1.8, 1])
            with col_b2:
                btn_batch = st.button("📂 Proses Batch File", key="btn_batch", use_container_width=True)
                
            if btn_batch:
                with st.spinner("Deep Learning sedang memproses data..."):
                    df['cleaned_text'] = df[col_name].apply(clean_text).apply(normalisasi_alay)
                    predictions = df['cleaned_text'].apply(predict_sentiment)
                    
                    df['predicted_sentiment'] = [p[0] for p in predictions]
                    df['confidence'] = [round(p[1] * 100, 2) for p in predictions]
                    
                    counts = df['predicted_sentiment'].value_counts()
                    bull_count = int(counts.get('bullying', 0))
                    non_bull_count = int(counts.get('non-bullying', 0))
                    
                    summary_text = f"Total Data: {len(df)} | Bullying: {bull_count} | Non-Bullying: {non_bull_count}"
                    
                    st.markdown("---")
                    
                    col_chart, col_stat = st.columns([1, 1])
                    with col_chart:
                        chart_data = pd.DataFrame({
                            'Kategori': ['Bullying', 'Non-Bullying'],
                            'Jumlah': [bull_count, non_bull_count]
                        }).set_index('Kategori')
                        st.bar_chart(chart_data)
                        
                    with col_stat:
                        st.markdown(f"""
                        <div class="text-center p-3">
                            <h4 class="fw-bold" style="color: #1e2640;">Statistik Batch IndoBERT</h4>
                            <p class="lead fw-semibold" style="color: #405590; font-size: 16px;">{summary_text}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="💾 Download CSV Berlabel",
                            data=csv_data,
                            file_name=f"hasil_analisis_{uploaded_file.name}",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    st.markdown("<h5 class='fw-bold mt-4 mb-2' style='color: #1e2640;'>Hasil Laporan Prediksi Detail</h5>", unsafe_allow_html=True)
                    formatted_df = df[[col_name, 'predicted_sentiment', 'confidence']].copy()
                    formatted_df.columns = ['Komentar', 'Sentiment', 'Confidence (%)']
                    st.dataframe(formatted_df, use_container_width=True)

# 8. FOOTER
st.markdown("<hr><p class='text-center text-muted small'>Copyright © 2026 David Rian Prabowo</p>", unsafe_allow_html=True)
