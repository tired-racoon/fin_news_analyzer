import re
import json
import numpy as np
import torch
import joblib
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="FinNews Analyzer",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

.stApp {
    background-color: #0a0a0f;
}

h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -0.02em;
}

.header-block {
    border-left: 3px solid #00ff88;
    padding: 0.4rem 1rem;
    margin-bottom: 2rem;
}

.header-block h1 {
    font-size: 2rem;
    color: #ffffff;
    margin: 0;
}

.header-block p {
    color: #666680;
    font-size: 0.85rem;
    margin: 0.2rem 0 0 0;
    font-family: 'IBM Plex Mono', monospace;
}

.metric-card {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.75rem;
}

.metric-card .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #555570;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}

.metric-card .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
}

.positive  { color: #00ff88; }
.negative  { color: #ff4466; }
.neutral   { color: #ffaa00; }

.ticker-chip {
    display: inline-block;
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 3px;
    padding: 0.2rem 0.6rem;
    margin: 0.2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #a0a0c8;
}

.prob-bar-wrap {
    margin: 0.3rem 0;
}

.prob-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #888899;
    display: flex;
    justify-content: space-between;
    margin-bottom: 2px;
}

.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #444460;
    border-bottom: 1px solid #1a1a2a;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
}

.conv-badge {
    background: #0a0a0f;
    border: 1px solid #00ff8844;
    color: #00ff8899;
    border-radius: 2px;
    padding: 0.15rem 0.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
}

.stTextArea textarea {
    background-color: #10101a !important;
    color: #e0e0f0 !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.95rem !important;
}

.stTextArea textarea:focus {
    border-color: #00ff8855 !important;
    box-shadow: 0 0 0 1px #00ff8833 !important;
}

.stButton button {
    background: #00ff88 !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 2rem !important;
    transition: opacity 0.15s !important;
}

.stButton button:hover {
    opacity: 0.85 !important;
}

.info-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 0.5rem;
}

.info-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #555570;
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 2px;
    padding: 0.2rem 0.6rem;
}

hr {
    border-color: #1a1a2a !important;
}

.stSpinner > div {
    border-top-color: #00ff88 !important;
}
</style>
""", unsafe_allow_html=True)

LABEL_MAP = {'negative': 0, 'neutral': 1, 'positive': 2}
ID2LABEL  = {0: 'negative', 1: 'neutral', 2: 'positive'}

TICKERS = [
    'SBER', 'LKOH', 'GAZP', 'ROSN', 'NVTK', 'GMKN', 'MGNT', 'TATN',
    'YNDX', 'MTSS', 'PLZL', 'CHMF', 'MAGN', 'ALRS', 'PHOR',
    'SNGS', 'SNGSP', 'TCSG', 'FIVE', 'OZON',
]

TICKER_KEYWORDS = {
    'SBER':  ['сбер', 'сбербанк'],
    'LKOH':  ['лукойл', 'lukoil'],
    'GAZP':  ['газпром', 'gazprom'],
    'ROSN':  ['роснефть', 'rosneft'],
    'NVTK':  ['новатэк', 'novatek'],
    'GMKN':  ['норникель', 'nornickel', 'гмк'],
    'MGNT':  ['магнит', 'magnit'],
    'TATN':  ['татнефть', 'tatneft'],
    'YNDX':  ['яндекс', 'yandex'],
    'MTSS':  ['мтс', ' mts '],
    'PLZL':  ['полюс', 'polyus'],
    'CHMF':  ['северсталь', 'severstal'],
    'MAGN':  ['ммк', 'магнитогорск'],
    'ALRS':  ['алроса', 'alrosa'],
    'PHOR':  ['фосагро', 'phosagro'],
    'SNGS':  ['сургутнефтегаз', 'surgutneftegas'],
    'SNGSP': ['сургутнефтегаз', 'surgutneftegas'],
    'TCSG':  ['тинькофф', 'тбанк', 'tinkoff', 't-bank'],
    'FIVE':  ['x5', 'пятёрочка', 'перекрёсток', 'карусель'],
    'OZON':  ['озон', 'ozon'],
}

OIL_TICKERS   = ['LKOH', 'ROSN', 'NVTK', 'TATN', 'SNGS', 'SNGSP', 'GAZP']
METAL_TICKERS = ['GMKN', 'PLZL', 'CHMF', 'MAGN', 'ALRS']

MACRO_KEYWORDS = {
    'oil':       (['нефть', 'brent', 'crude', 'баррель', 'опек', 'opec'], OIL_TICKERS),
    'metals':    (['металл', 'сталь', 'никель', 'палладий', 'алмаз'], METAL_TICKERS),
    'ruble':     (['рубль', 'курс доллара', 'курс евро', 'девальвац', 'цб рф', 'банк росс'], TICKERS),
    'sanctions': (['санкци', 'embargo', 'эмбарго'], TICKERS),
    'rate':      (['ключевая ставка', 'процентная ставка', 'цб повыс', 'цб снизи'], TICKERS),
}

MAX_LEN      = 512
TICKER_MAX_LEN = 256
DEVICE       = torch.device('cpu')
BEST_THRESH  = 0.38


def get_relevant_tickers_kw(text):
    text_lower = str(text).lower()
    relevant   = set()
    for ticker, keywords in TICKER_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                relevant.add(ticker)
                break
    for _, (kws, affected) in MACRO_KEYWORDS.items():
        for kw in kws:
            if kw in text_lower:
                for t in affected:
                    relevant.add(t)
                break
    return sorted(relevant)


@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    tok   = AutoTokenizer.from_pretrained('models/sentiment_best')
    model = AutoModelForSequenceClassification.from_pretrained('models/sentiment_best')
    model.eval()
    return tok, model


@st.cache_resource(show_spinner=False)
def load_ticker_model():
    tok   = AutoTokenizer.from_pretrained('models/ticker_relevance')
    model = AutoModelForSequenceClassification.from_pretrained('models/ticker_relevance')
    model.eval()
    return tok, model


@st.cache_resource(show_spinner=False)
def load_conv_model():
    try:
        from conv_model import conv_model as cmodel
        return cmodel
    except Exception:
        return None


def predict_sentiment(text, tok, model):
    enc = tok(
        text, truncation=True, padding=True,
        max_length=MAX_LEN, return_tensors='pt',
    )
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()[0]
    label_id = int(np.argmax(probs))
    return ID2LABEL[label_id], probs


def predict_tickers_ml(text, tok, model, threshold=BEST_THRESH):
    enc = tok(
        text, truncation=True, padding=True,
        max_length=TICKER_MAX_LEN, return_tensors='pt',
    )
    with torch.no_grad():
        probs = torch.sigmoid(model(**enc).logits).cpu().numpy()[0]
    return sorted([TICKERS[i] for i, p in enumerate(probs) if p >= threshold]), probs


def get_relevant_tickers_combined(text, tok_t, model_t, threshold=BEST_THRESH):
    kw_tickers = set(get_relevant_tickers_kw(text))
    ml_tickers, _ = predict_tickers_ml(text, tok_t, model_t, threshold)
    return sorted(set(ml_tickers) | kw_tickers)


def conv_predict(text, conv_model_obj):
    """Прогноз старой модели на котировках (условный — возвращает dummy если нет данных)."""
    try:
        import numpy as np
        tickers_found = get_relevant_tickers_kw(text)
        if not tickers_found or conv_model_obj is None:
            return None
        dummy_input = np.zeros((1, 20))
        raw = conv_model_obj.predict(dummy_input)
        return float(raw[0][0])
    except Exception:
        return None


def render_prob_bars(probs):
    labels  = ['negative', 'neutral', 'positive']
    colors  = ['#ff4466', '#ffaa00', '#00ff88']
    bars_html = ""
    for lbl, p, col in zip(labels, probs, colors):
        pct = int(p * 100)
        bars_html += f"""
        <div class="prob-bar-wrap">
            <div class="prob-label"><span>{lbl}</span><span>{pct}%</span></div>
            <div style="height:4px;background:#1a1a2a;border-radius:2px;">
                <div style="height:4px;width:{pct}%;background:{col};border-radius:2px;transition:width 0.4s;"></div>
            </div>
        </div>"""
    return bars_html


st.markdown("""
<div class="header-block">
    <h1>FinNews Analyzer</h1>
    <p>// анализ сентимента русских финансовых новостей · Мосбиржа top-20</p>
</div>
""", unsafe_allow_html=True)

col_input, col_results = st.columns([1, 1], gap="large")

with col_input:
    st.markdown('<div class="section-title">Текст новости</div>', unsafe_allow_html=True)

    news_text = st.text_area(
        label="",
        placeholder="Вставьте заголовок и текст финансовой новости...",
        height=220,
        label_visibility="collapsed",
    )

    col_btn, col_thresh = st.columns([1, 1])
    with col_btn:
        run_btn = st.button("АНАЛИЗИРОВАТЬ →", use_container_width=True)
    with col_thresh:
        threshold = st.slider(
            "Порог ML (тикеры)",
            min_value=0.1, max_value=0.8,
            value=BEST_THRESH, step=0.05,
            help="Минимальная уверенность ML-модели для включения тикера",
        )

    st.markdown('<div class="section-title">Примеры</div>', unsafe_allow_html=True)
    examples = [
        "Газпром объявил о рекордных дивидендах за 2023 год, акционеры получат 50 рублей на акцию.",
        "Центральный банк России повысил ключевую ставку до 21%, что усилило давление на фондовый рынок.",
        "Цены на нефть Brent упали ниже 70 долларов на фоне решения ОПЕК+ увеличить добычу.",
        "Сбербанк сообщил о рекордной прибыли по итогам квартала, превысившей ожидания аналитиков.",
    ]
    for ex in examples:
        if st.button(ex[:60] + "…", key=ex, use_container_width=True):
            st.session_state['example_text'] = ex
            st.rerun()

if 'example_text' in st.session_state:
    news_text = st.session_state.pop('example_text')
    run_btn   = True

with col_results:
    st.markdown('<div class="section-title">Результаты</div>', unsafe_allow_html=True)

    if run_btn and news_text.strip():
        with st.spinner(""):
            try:
                tok_s, model_s = load_sentiment_model()
                tok_t, model_t = load_ticker_model()
            except Exception as e:
                st.error(f"Не удалось загрузить модели: {e}")
                st.stop()

            label, probs = predict_sentiment(news_text, tok_s, model_s)
            tickers_combined = get_relevant_tickers_combined(news_text, tok_t, model_t, threshold)
            tickers_ml, ticker_probs = predict_tickers_ml(news_text, tok_t, model_t, threshold)
            tickers_kw = get_relevant_tickers_kw(news_text)

        label_class = label
        label_emoji = {'positive': '↑', 'negative': '↓', 'neutral': '→'}[label]

        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Сентимент · ruBERT-base</div>
            <div class="value {label_class}">{label_emoji} {label.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Вероятности классов</div>
            {render_prob_bars(probs)}
        </div>
        """, unsafe_allow_html=True)

        if tickers_combined:
            chips = "".join(f'<span class="ticker-chip">{t}</span>' for t in tickers_combined)
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Затронутые тикеры · combined (ML + keyword)</div>
                <div style="margin-top:0.5rem">{chips}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <div class="label">Затронутые тикеры</div>
                <div style="color:#555570;font-family:'IBM Plex Mono',monospace;font-size:0.85rem;margin-top:0.4rem">— тикеры не определены</div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Детали: ML vs Keyword"):
            ml_chips  = "".join(f'<span class="ticker-chip">{t}</span>' for t in tickers_ml)  or "—"
            kw_chips  = "".join(f'<span class="ticker-chip">{t}</span>' for t in tickers_kw) or "—"
            st.markdown(f"""
            <div style="margin-bottom:0.8rem">
                <div class="label" style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#555570;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem">ML-модель (rubert-tiny2 multi-label, порог {threshold:.2f})</div>
                {ml_chips}
            </div>
            <div>
                <div class="label" style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#555570;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem">Keyword / macro matching</div>
                {kw_chips}
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div style="margin-top:1rem"><div class="label" style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;color:#555570;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem">Вероятности ML по всем тикерам</div></div>', unsafe_allow_html=True)
            ticker_prob_pairs = sorted(zip(TICKERS, ticker_probs), key=lambda x: x[1], reverse=True)
            rows_html = ""
            for t, p in ticker_prob_pairs:
                pct  = int(p * 100)
                col  = '#00ff88' if p >= threshold else '#2a2a4a'
                rows_html += f"""
                <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.25rem;">
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#888899;width:3.5rem">{t}</span>
                    <div style="flex:1;height:3px;background:#1a1a2a;border-radius:2px;">
                        <div style="height:3px;width:{pct}%;background:{col};border-radius:2px;"></div>
                    </div>
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#555570;width:2.5rem;text-align:right">{p:.2f}</span>
                </div>"""
            st.markdown(rows_html, unsafe_allow_html=True)

        conv_model_obj = load_conv_model()
        if conv_model_obj is not None:
            st.markdown('<div class="section-title">Прогноз старой модели (котировки)</div>', unsafe_allow_html=True)
            score = conv_predict(news_text, conv_model_obj)
            if score is not None:
                direction = '↑ positive' if score > 10 else ('↓ negative' if score < -10 else '→ neutral')
                dir_class = 'positive' if score > 10 else ('negative' if score < -10 else 'neutral')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">conv_model · прогноз изменения котировок</div>
                    <div class="value {dir_class}">{direction}</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#555570;margin-top:0.3rem">raw score: {score:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <div class="label">conv_model · прогноз изменения котировок</div>
                    <div style="color:#555570;font-family:'IBM Plex Mono',monospace;font-size:0.82rem;margin-top:0.4rem">Нет свежих данных котировок для прогноза</div>
                </div>
                """, unsafe_allow_html=True)

    elif run_btn:
        st.markdown("""
        <div style="color:#444460;font-family:'IBM Plex Mono',monospace;font-size:0.85rem;padding:2rem 0">
            // введите текст новости
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="color:#333350;font-family:'IBM Plex Mono',monospace;font-size:0.85rem;padding:2rem 0;line-height:1.8">
            // ожидание ввода<br>
            // введите текст и нажмите АНАЛИЗИРОВАТЬ
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#333350;text-align:center;padding:0.5rem 0">
    FinNews Analyzer · sentiment: ruBERT-base (F1=0.71) · tickers: rubert-tiny2 multi-label + focal loss · keyword/macro matching
</div>
""", unsafe_allow_html=True)