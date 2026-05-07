import numpy as np
import streamlit as st
from PIL import Image
from predict import preprocess, predict

st.set_page_config(page_title="MNIST Digit Recogniser", page_icon="🔢", layout="centered")

st.title("🔢 Handwritten Digit Recogniser")
st.caption("CNN built from scratch in NumPy — no PyTorch, no TensorFlow")

CONFIDENCE_THRESHOLD = 0.60

# ── Input mode ───────────────────────────────────────────────────────────────
mode = st.radio("Input method", ["Draw a digit", "Upload an image"], horizontal=True)

pil_img = None

if mode == "Draw a digit":
    from streamlit_drawable_canvas import st_canvas
    st.markdown("Draw a single digit (0–9) in the box below.")
    canvas = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas.image_data is not None:
        # canvas gives RGBA uint8 — convert to PIL
        pil_img = Image.fromarray(canvas.image_data.astype(np.uint8))

else:
    uploaded = st.file_uploader("Upload a 28×28 digit image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded:
        pil_img = Image.open(uploaded)

# ── Inference ────────────────────────────────────────────────────────────────
if pil_img is not None:
    processed, thumb = preprocess(pil_img)

    # Skip all-black canvas (nothing drawn yet)
    if thumb.max() < 10:
        st.info("Draw a digit above to see the prediction.")
        st.stop()

    probs = predict(processed)
    pred  = int(np.argmax(probs))
    conf  = float(probs[pred])

    # ── Layout: original | what model sees ───────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Your input**")
        st.image(pil_img, width=140)
    with col2:
        st.markdown("**What the model sees** (28×28)")
        st.image(Image.fromarray(thumb), width=140)

    st.divider()

    # ── Prediction result ─────────────────────────────────────────────────────
    if conf >= CONFIDENCE_THRESHOLD:
        st.markdown(
            f"<h1 style='text-align:center; color:#1A56A8;'>Predicted: {pred}</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='text-align:center; font-size:18px;'>Confidence: <b>{conf*100:.1f}%</b></p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<h1 style='text-align:center; color:#e07b00;'>Predicted: {pred}</h1>",
            unsafe_allow_html=True
        )
        st.warning(
            f"Low confidence ({conf*100:.1f}%) — the model is uncertain. "
            "Try drawing more clearly or centring the digit."
        )

    st.divider()

    # ── Probability bar chart ─────────────────────────────────────────────────
    st.markdown("**Class probabilities**")
    import pandas as pd
    chart_data = pd.DataFrame({
        "Digit":       [str(i) for i in range(10)],
        "Probability": probs * 100
    }).set_index("Digit")
    st.bar_chart(chart_data, y="Probability", color="#1A56A8")

    # ── Raw breakdown ─────────────────────────────────────────────────────────
    with st.expander("Full probability breakdown"):
        for i, p in enumerate(probs):
            marker = " ← predicted" if i == pred else ""
            st.text(f"  {i}: {p*100:6.2f}%{marker}")