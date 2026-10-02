import base64
import html
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import cv2
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Video-LLM Classroom Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOCAL + STREAMLIT CLOUD CONFIGURATION
# ============================================================

load_dotenv()


def get_config(name, default=None):
    """
    Read configuration from Streamlit Cloud Secrets first.

    If the application is running locally, fall back to
    environment variables loaded from the local .env file.
    """

    try:
        if name in st.secrets:
            value = st.secrets[name]

            if isinstance(value, str):
                return value.strip()

            return value

    except Exception:
        pass

    value = os.getenv(name, default)

    if isinstance(value, str):
        return value.strip()

    return value


OPENAI_API_KEY = get_config("OPENAI_API_KEY")

OPENAI_MODEL = get_config(
    "OPENAI_MODEL",
    "gpt-6-luna",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(68, 98, 255, 0.14),
            transparent 32%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(157, 78, 221, 0.12),
            transparent 28%
        ),
        #07101f;

    color: #f4f7fb;
}

[data-testid="stAppViewContainer"] {
    background: transparent;
}

[data-testid="stHeader"] {
    background: rgba(7, 16, 31, 0.76);
    backdrop-filter: blur(12px);
}

.block-container {
    max-width: 1480px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* SIDEBAR */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            rgba(10, 19, 38, 0.98),
            rgba(8, 15, 30, 0.98)
        );

    border-right:
        1px solid rgba(118, 145, 255, 0.14);
}

[data-testid="stSidebar"] * {
    color: #e8eefc;
}

.sidebar-brand {
    font-size: 1.35rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
}

.sidebar-subtitle {
    color: #7f91ad;
    font-size: 0.82rem;
    margin-bottom: 1.5rem;
}

.sidebar-card {
    padding: 0.95rem 1rem;
    margin-bottom: 0.8rem;
    border-radius: 14px;

    background:
        rgba(255, 255, 255, 0.035);

    border:
        1px solid rgba(122, 150, 255, 0.10);
}

.sidebar-label {
    color: #7690b9;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.25rem;
}

.sidebar-value {
    color: #ffffff;
    font-weight: 600;
    font-size: 0.88rem;
}

.nav-item {
    padding: 0.65rem 0.75rem;
    border-radius: 10px;
    margin: 0.20rem 0;
    color: #aebbd0;
    font-size: 0.88rem;
}


/* HERO */

.hero {
    position: relative;
    overflow: hidden;

    border-radius: 28px;

    padding: 2.8rem 3rem;

    margin-bottom: 2rem;

    background:
        linear-gradient(
            135deg,
            rgba(18, 31, 59, 0.96),
            rgba(13, 24, 47, 0.90)
        );

    border:
        1px solid rgba(110, 137, 255, 0.18);

    box-shadow:
        0 22px 60px rgba(0, 0, 0, 0.30);
}

.hero:before {
    content: "";

    position: absolute;

    width: 440px;
    height: 440px;

    right: -140px;
    top: -170px;

    background:
        radial-gradient(
            circle,
            rgba(74, 123, 255, 0.22),
            rgba(129, 71, 255, 0.05),
            transparent 72%
        );
}

.hero-kicker {
    color: #72a7ff;
    font-size: 0.75rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 0.7rem;
}

.hero-title {
    font-size: 3.25rem;
    line-height: 1.02;
    font-weight: 850;
    letter-spacing: -0.045em;
}

.hero-gradient {
    background:
        linear-gradient(
            90deg,
            #f7fbff,
            #9fc1ff,
            #c5a4ff
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #8ea2c4;
    margin-top: 1rem;
    font-size: 1rem;
    line-height: 1.65;
    max-width: 800px;
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 1.4rem;
}

.badge {
    padding: 0.46rem 0.78rem;
    border-radius: 999px;

    font-size: 0.72rem;
    font-weight: 700;

    background:
        rgba(91, 132, 255, 0.08);

    border:
        1px solid rgba(104, 145, 255, 0.22);

    color: #bcd0ff;
}


/* SECTION HEADERS */

.section-header {
    margin-top: 2.2rem;
    margin-bottom: 1rem;
}

.section-index {
    color: #6e9bff;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.section-title {
    font-size: 1.65rem;
    font-weight: 760;
    letter-spacing: -0.025em;
    margin-top: 0.18rem;
}

.section-description {
    color: #7f91ad;
    font-size: 0.9rem;
    max-width: 880px;
}


/* CARDS */

.glass-card {
    background: rgba(16, 28, 53, 0.76);

    border:
        1px solid rgba(111, 143, 255, 0.13);

    border-radius: 19px;

    padding: 1.25rem;

    box-shadow:
        0 16px 38px rgba(0, 0, 0, 0.22);
}

.metric-card {
    min-height: 116px;

    border-radius: 18px;

    padding: 1.15rem 1.25rem;

    background:
        linear-gradient(
            145deg,
            rgba(17, 31, 60, 0.94),
            rgba(12, 24, 48, 0.92)
        );

    border:
        1px solid rgba(110, 143, 255, 0.15);

    box-shadow:
        0 14px 32px rgba(0, 0, 0, 0.18);
}

.metric-label {
    color: #6f83a5;

    font-size: 0.67rem;

    letter-spacing: 0.12em;

    text-transform: uppercase;

    margin-bottom: 0.5rem;

    font-weight: 700;
}

.metric-value {
    color: #ffffff;

    font-weight: 760;

    font-size: 1.55rem;
}

.metric-sub {
    color: #6582b3;
    font-size: 0.76rem;
    margin-top: 0.25rem;
}

.ai-card {
    background:
        linear-gradient(
            145deg,
            rgba(15, 29, 56, 0.96),
            rgba(12, 23, 44, 0.94)
        );

    border:
        1px solid rgba(95, 130, 255, 0.14);

    border-radius: 18px;

    padding: 1.25rem;

    min-height: 175px;
}

.ai-label {
    color: #71a2ff;

    font-size: 0.67rem;

    letter-spacing: 0.13em;

    text-transform: uppercase;

    font-weight: 750;

    margin-bottom: 0.65rem;
}

.ai-text {
    color: #cbd7ed;
    font-size: 0.9rem;
    line-height: 1.7;
}

.ai-placeholder {
    color: #7687a4;
    font-size: 0.88rem;
    line-height: 1.6;
}


/* FRAME CARDS */

.frame-top {
    display: flex;
    justify-content: space-between;
    align-items: center;

    margin-bottom: 0.5rem;
}

.frame-number {
    font-size: 0.68rem;
    color: #9ab8ff;
    letter-spacing: 0.1em;
    font-weight: 750;
}

.timestamp {
    font-size: 0.67rem;
    color: #5ed6ff;

    background:
        rgba(44, 195, 255, 0.08);

    border:
        1px solid rgba(44, 195, 255, 0.16);

    padding: 0.2rem 0.45rem;

    border-radius: 999px;
}


/* PIPELINE */

.pipeline {
    display: grid;

    grid-template-columns:
        minmax(130px, 1fr)
        auto
        minmax(130px, 1fr)
        auto
        minmax(130px, 1fr)
        auto
        minmax(130px, 1fr)
        auto
        minmax(130px, 1fr);

    align-items: center;

    gap: 0.7rem;

    margin-top: 1rem;
}

.pipeline-node {
    border-radius: 16px;

    padding: 1rem 0.75rem;

    text-align: center;

    background:
        linear-gradient(
            145deg,
            rgba(15, 31, 61, 0.96),
            rgba(12, 25, 49, 0.92)
        );

    border:
        1px solid rgba(97, 136, 255, 0.15);
}

.pipeline-node-title {
    color: #dce7ff;
    font-weight: 700;
    font-size: 0.78rem;
}

.pipeline-node-sub {
    color: #6980a6;
    font-size: 0.65rem;
    margin-top: 0.25rem;
}

.pipeline-arrow {
    text-align: center;
    color: #6f91ff;
    font-size: 1.1rem;
}


/* CONTROLS */

div[data-testid="stFileUploader"] section {
    background:
        rgba(17, 30, 55, 0.70);

    border:
        1px dashed rgba(104, 144, 255, 0.30);

    border-radius: 18px;

    padding: 1.3rem;
}

.stButton > button {
    border-radius: 12px;

    min-height: 46px;

    background:
        linear-gradient(
            90deg,
            #4169e1,
            #644ee8
        );

    color: white;

    font-weight: 700;

    border:
        1px solid rgba(150, 170, 255, 0.22);

    box-shadow:
        0 10px 24px rgba(69, 82, 229, 0.18);
}

.stButton > button:hover {
    border-color:
        rgba(160, 181, 255, 0.38);
}

div[data-testid="stTextInput"] input {
    background:
        rgba(15, 28, 51, 0.78);

    color: #f2f6ff;

    border:
        1px solid rgba(105, 137, 255, 0.18);

    border-radius: 12px;

    min-height: 46px;
}

.question-chip {
    display: inline-block;

    margin:
        0.25rem 0.35rem 0.25rem 0;

    padding:
        0.48rem 0.7rem;

    border-radius: 999px;

    color: #a9bce1;

    background:
        rgba(96, 126, 220, 0.055);

    border:
        1px solid rgba(100, 137, 244, 0.12);

    font-size: 0.73rem;
}

.answer-box {
    margin-top: 1rem;

    padding: 1.25rem;

    border-radius: 16px;

    background:
        rgba(15, 31, 61, 0.82);

    border:
        1px solid rgba(95, 138, 255, 0.16);

    color: #d6e1f7;

    line-height: 1.65;
}

.footer {
    text-align: center;

    padding-top: 2.7rem;
    padding-bottom: 1rem;

    color: #52647f;

    font-size: 0.76rem;
}


@media (max-width: 1100px) {

    .pipeline {
        grid-template-columns: 1fr;
    }

    .pipeline-arrow {
        transform: rotate(90deg);
    }

    .hero-title {
        font-size: 2.4rem;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def safe_text(value):
    """
    Escape AI-generated text before inserting it into HTML.
    """

    if value is None:
        return ""

    return html.escape(str(value)).replace("\n", "<br>")


def render_section(index_text, title, description=""):
    st.html(
        f"""
<div class="section-header">
    <div class="section-index">
        {safe_text(index_text)}
    </div>

    <div class="section-title">
        {safe_text(title)}
    </div>

    <div class="section-description">
        {safe_text(description)}
    </div>
</div>
"""
    )


def render_ai_card(label, content, placeholder=False):
    css_class = (
        "ai-placeholder"
        if placeholder
        else "ai-text"
    )

    st.html(
        f"""
<div class="ai-card">
    <div class="ai-label">
        {safe_text(label)}
    </div>

    <div class="{css_class}">
        {safe_text(content)}
    </div>
</div>
"""
    )


# ============================================================
# OPENAI
# ============================================================

def get_openai_client():

    if not OPENAI_API_KEY:
        return None

    return OpenAI(
        api_key=OPENAI_API_KEY
    )


# ============================================================
# IMAGE ENCODING
# ============================================================

def image_to_data_url(image_path):

    with open(
        image_path,
        "rb",
    ) as image_file:

        encoded = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


# ============================================================
# VIDEO PROCESSING
# ============================================================

def get_video_metadata(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise ValueError(
            "Could not open the uploaded video."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    duration = (
        frame_count / fps
        if fps > 0
        else 0
    )

    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
    }


def extract_frames(
    video_path,
    output_dir,
    num_frames=8,
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for old_file in output_dir.glob(
        "*.jpg"
    ):
        try:
            old_file.unlink()
        except OSError:
            pass

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise ValueError(
            "Unable to read the uploaded video."
        )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if frame_count <= 0:

        cap.release()

        raise ValueError(
            "Video contains no readable frames."
        )

    num_frames = min(
        num_frames,
        frame_count,
    )

    if num_frames == 1:

        indices = [0]

    else:

        indices = [
            int(
                i
                * (frame_count - 1)
                / (num_frames - 1)
            )
            for i in range(num_frames)
        ]

    results = []

    for i, frame_index in enumerate(
        indices,
        start=1,
    ):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index,
        )

        success, frame = cap.read()

        if not success:
            continue

        frame_path = (
            output_dir
            / f"frame_{i:02d}.jpg"
        )

        cv2.imwrite(
            str(frame_path),
            frame,
        )

        timestamp = (
            frame_index / fps
            if fps > 0
            else 0
        )

        results.append(
            {
                "path":
                    str(frame_path),

                "timestamp":
                    timestamp,

                "frame_index":
                    frame_index,
            }
        )

    cap.release()

    return results


def format_timestamp(seconds):

    minutes = int(
        seconds // 60
    )

    remaining_seconds = (
        seconds % 60
    )

    return (
        f"{minutes:02d}:"
        f"{remaining_seconds:04.1f}"
    )


# ============================================================
# MULTIMODAL API REQUEST
# ============================================================

def send_frames_to_model(
    frames,
    prompt,
):

    client = get_openai_client()

    if client is None:

        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    if not frames:

        raise RuntimeError(
            "No sampled video frames are available."
        )

    content = [
        {
            "type": "input_text",
            "text": prompt,
        }
    ]

    for number, frame in enumerate(
        frames,
        start=1,
    ):

        timestamp = format_timestamp(
            frame["timestamp"]
        )

        content.append(
            {
                "type": "input_text",

                "text":
                    f"Chronological frame {number} "
                    f"at timestamp {timestamp}.",
            }
        )

        content.append(
            {
                "type": "input_image",

                "image_url":
                    image_to_data_url(
                        frame["path"]
                    ),

                "detail":
                    "low",
            }
        )

    response = client.responses.create(
        model=OPENAI_MODEL,

        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
    )

    return response.output_text


# ============================================================
# VIDEO ANALYSIS
# ============================================================

def extract_json(text):

    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if (
        start != -1
        and end != -1
        and end > start
    ):

        try:
            return json.loads(
                cleaned[start:end + 1]
            )

        except json.JSONDecodeError:
            pass

    return None


def analyze_video_with_llm(frames):

    prompt = """
You are analyzing chronological frames sampled uniformly
from one continuous video.

The supplied images appear in chronological order.

Use only information supported by the supplied frames.

Do not invent actions, objects, dialogue, audio, or events
that are not reasonably visible.

Because these are sampled frames and not every frame from
the video, acknowledge uncertainty when appropriate.

Analyze the video for a Computer Vision feasibility study.

Return ONLY valid JSON with exactly this structure:

{
  "summary": "Concise video summary",
  "objects": [
    "object 1",
    "object 2"
  ],
  "actions": [
    "action 1",
    "action 2"
  ],
  "temporal_sequence": [
    "First ...",
    "Then ...",
    "Finally ..."
  ]
}

summary:
Describe the overall video.

objects:
List important visible objects.

actions:
List the major observable actions.

temporal_sequence:
Describe the likely chronological order of major events.

Return JSON only.
"""

    result = send_frames_to_model(
        frames,
        prompt,
    )

    parsed = extract_json(
        result
    )

    if parsed is None:

        return {
            "summary": result,
            "objects": [],
            "actions": [],
            "temporal_sequence": [],
        }

    return parsed


# ============================================================
# QUESTION ANSWERING
# ============================================================

def ask_video_question(
    frames,
    question,
):

    prompt = f"""
You are answering a question about one continuous video.

The supplied images are chronological frames sampled
uniformly from that video.

Answer using only evidence visible in those frames.

Rules:

1. Do not invent events.
2. Consider chronological frame order.
3. For before-and-after questions, use temporal ordering.
4. If evidence is insufficient, clearly say so.
5. Keep the answer concise but explanatory.

Question:

{question}
"""

    return send_frames_to_model(
        frames,
        prompt,
    )


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "frames": [],
    "metadata": None,
    "temp_video_path": None,
    "frames_dir": None,
    "last_upload_id": None,
    "analysis": None,
    "last_answer": None,
    "feasibility_answers": {},
}


for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
<div class="sidebar-brand">
    Video-LLM Lab
</div>

<div class="sidebar-subtitle">
    Multimodal video intelligence prototype
</div>
"""
    )

    api_status = (
        "Configured"
        if OPENAI_API_KEY
        else "Not configured"
    )

    st.html(
        f"""
<div class="sidebar-card">

    <div class="sidebar-label">
        Project
    </div>

    <div class="sidebar-value">
        Video-LLM Classroom Assistant
    </div>

</div>


<div class="sidebar-card">

    <div class="sidebar-label">
        Challenge
    </div>

    <div class="sidebar-value">
        CVPR 2025 Video-LLM Workshop
    </div>

</div>


<div class="sidebar-card">

    <div class="sidebar-label">
        Pipeline
    </div>

    <div class="sidebar-value">
        Video -> Frames -> LLM -> Reasoning
    </div>

</div>


<div class="sidebar-card">

    <div class="sidebar-label">
        AI Model
    </div>

    <div class="sidebar-value">
        {safe_text(OPENAI_MODEL)}
    </div>

</div>


<div class="sidebar-card">

    <div class="sidebar-label">
        API Status
    </div>

    <div class="sidebar-value">
        {safe_text(api_status)}
    </div>

</div>
"""
    )

    st.markdown("---")

    st.html(
        """
<div class="nav-item">
    01&nbsp;&nbsp; Video Input
</div>

<div class="nav-item">
    02&nbsp;&nbsp; Temporal Sampling
</div>

<div class="nav-item">
    03&nbsp;&nbsp; Video Intelligence
</div>

<div class="nav-item">
    04&nbsp;&nbsp; Ask Video
</div>

<div class="nav-item">
    05&nbsp;&nbsp; Feasibility Test
</div>
"""
    )


# ============================================================
# HERO
# ============================================================

st.html(
    """
<div class="hero">

    <div class="hero-kicker">
        CVPR 2025 VIDEO-LLM CHALLENGE
    </div>

    <div class="hero-title">

        <span class="hero-gradient">
            Classroom Intelligence
        </span>

    </div>

    <div class="hero-subtitle">

        An experimental multimodal Video-LLM system designed
        to understand visual events, reason about temporal
        relationships, summarize video content, and support
        natural-language interaction.

    </div>

    <div class="badge-row">

        <div class="badge">
            CVPR 2025
        </div>

        <div class="badge">
            VIDEO UNDERSTANDING
        </div>

        <div class="badge">
            MULTIMODAL AI
        </div>

        <div class="badge">
            TEMPORAL REASONING
        </div>

    </div>

</div>
"""
)


# ============================================================
# API STATUS WARNING
# ============================================================

if not OPENAI_API_KEY:

    st.error(
        "OpenAI API key was not detected. "
        "For local use, add OPENAI_API_KEY to .env. "
        "For Streamlit Cloud, add OPENAI_API_KEY "
        "under Manage app -> Settings -> Secrets."
    )


# ============================================================
# VIDEO INPUT
# ============================================================

render_section(
    "01 / VIDEO INPUT",
    "Upload Video",
    (
        "Upload a short instructional or activity video. "
        "The system extracts metadata and representative "
        "frames for multimodal analysis."
    ),
)


uploaded_file = st.file_uploader(
    "Upload MP4, MOV or AVI",
    type=[
        "mp4",
        "mov",
        "avi",
    ],
    label_visibility="collapsed",
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    upload_id = (
        uploaded_file.name,
        uploaded_file.size,
    )

    suffix = Path(
        uploaded_file.name
    ).suffix


    # --------------------------------------------------------
    # NEW VIDEO
    # --------------------------------------------------------

    if (
        st.session_state.last_upload_id
        != upload_id
    ):

        old_video = (
            st.session_state.temp_video_path
        )

        old_frames_dir = (
            st.session_state.frames_dir
        )

        if old_video:

            try:
                os.remove(old_video)
            except OSError:
                pass

        if old_frames_dir:

            try:
                shutil.rmtree(
                    old_frames_dir
                )
            except OSError:
                pass

        temp_file = (
            tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            )
        )

        temp_file.write(
            uploaded_file.getbuffer()
        )

        temp_file.close()

        frames_dir = tempfile.mkdtemp(
            prefix="video_llm_frames_"
        )

        st.session_state.temp_video_path = (
            temp_file.name
        )

        st.session_state.frames_dir = (
            frames_dir
        )

        st.session_state.last_upload_id = (
            upload_id
        )

        st.session_state.frames = []

        st.session_state.analysis = None

        st.session_state.last_answer = None

        st.session_state.feasibility_answers = {}

        # Clear old human evaluations.
        for key in list(
            st.session_state.keys()
        ):
            if key.startswith(
                "evaluation_"
            ):
                del st.session_state[key]

        try:

            st.session_state.metadata = (
                get_video_metadata(
                    st.session_state.temp_video_path
                )
            )

        except Exception as error:

            st.error(
                f"Could not read video metadata: "
                f"{error}"
            )

            st.session_state.metadata = None


    # ========================================================
    # VIDEO PREVIEW
    # ========================================================

    st.video(
        uploaded_file
    )

    metadata = (
        st.session_state.metadata
    )


    # ========================================================
    # METADATA
    # ========================================================

    if metadata:

        render_section(
            "VIDEO METADATA",
            "Video Information",
            (
                "Technical information extracted "
                "directly from the uploaded video "
                "using OpenCV."
            ),
        )


        col1, col2, col3, col4 = (
            st.columns(4)
        )


        with col1:

            st.html(
                f"""
<div class="metric-card">

    <div class="metric-label">
        Duration
    </div>

    <div class="metric-value">
        {metadata["duration"]:.2f} sec
    </div>

    <div class="metric-sub">
        Total playback time
    </div>

</div>
"""
            )


        with col2:

            st.html(
                f"""
<div class="metric-card">

    <div class="metric-label">
        Frame Rate
    </div>

    <div class="metric-value">
        {metadata["fps"]:.2f} FPS
    </div>

    <div class="metric-sub">
        Frames per second
    </div>

</div>
"""
            )


        with col3:

            st.html(
                f"""
<div class="metric-card">

    <div class="metric-label">
        Total Frames
    </div>

    <div class="metric-value">
        {metadata["frame_count"]}
    </div>

    <div class="metric-sub">
        Frames detected
    </div>

</div>
"""
            )


        with col4:

            st.html(
                f"""
<div class="metric-card">

    <div class="metric-label">
        Resolution
    </div>

    <div class="metric-value">
        {metadata["width"]} x {metadata["height"]}
    </div>

    <div class="metric-sub">
        Video dimensions
    </div>

</div>
"""
            )


        # ====================================================
        # TEMPORAL SAMPLING
        # ====================================================

        render_section(
            "02 / TEMPORAL SAMPLING",
            "Video Timeline",
            (
                "Eight frames are sampled uniformly "
                "across the video to provide spatial "
                "and temporal context."
            ),
        )


        if st.button(
            "Analyze Video with AI",
            type="primary",
        ):

            if not OPENAI_API_KEY:

                st.error(
                    "OPENAI_API_KEY is not configured."
                )

            else:

                try:

                    with st.spinner(
                        "Sampling video frames..."
                    ):

                        st.session_state.frames = (
                            extract_frames(
                                st.session_state.temp_video_path,
                                st.session_state.frames_dir,
                                num_frames=8,
                            )
                        )


                    if not st.session_state.frames:

                        raise RuntimeError(
                            "No video frames were extracted."
                        )


                    with st.spinner(
                        "Video-LLM is analyzing "
                        "objects, actions, and temporal order..."
                    ):

                        st.session_state.analysis = (
                            analyze_video_with_llm(
                                st.session_state.frames
                            )
                        )


                    st.success(
                        "Video analysis completed successfully."
                    )


                except Exception as error:

                    st.error(
                        f"Analysis failed: {error}"
                    )


        frames = (
            st.session_state.frames
        )


        # ====================================================
        # DISPLAY FRAMES
        # ====================================================

        if frames:

            for row_start in range(
                0,
                len(frames),
                4,
            ):

                row_frames = frames[
                    row_start:
                    row_start + 4
                ]

                columns = st.columns(4)


                for idx, (
                    column,
                    frame_info,
                ) in enumerate(
                    zip(
                        columns,
                        row_frames,
                    ),
                    start=row_start + 1,
                ):

                    with column:

                        timestamp = (
                            format_timestamp(
                                frame_info[
                                    "timestamp"
                                ]
                            )
                        )

                        st.html(
                            f"""
<div class="frame-top">

    <div class="frame-number">
        FRAME {idx:02d}
    </div>

    <div class="timestamp">
        {timestamp}
    </div>

</div>
"""
                        )

                        st.image(
                            frame_info["path"],
                            width="stretch",
                        )


        # ====================================================
        # VIDEO INTELLIGENCE
        # ====================================================

        render_section(
            "03 / VIDEO INTELLIGENCE",
            "Multimodal Analysis",
            (
                "AI-generated semantic and temporal "
                "understanding derived from the sampled "
                "video frames."
            ),
        )


        analysis = (
            st.session_state.analysis
        )


        if analysis:

            objects = analysis.get(
                "objects",
                [],
            )

            actions = analysis.get(
                "actions",
                [],
            )

            timeline = analysis.get(
                "temporal_sequence",
                [],
            )


            objects_text = (
                ", ".join(
                    str(item)
                    for item in objects
                )
                if objects
                else
                "No reliable objects identified."
            )


            actions_text = (
                "\n".join(
                    f"- {item}"
                    for item in actions
                )
                if actions
                else
                "No reliable actions identified."
            )


            temporal_text = (
                "\n".join(
                    f"{index}. {item}"
                    for index, item in enumerate(
                        timeline,
                        start=1,
                    )
                )
                if timeline
                else
                "Temporal sequence unavailable."
            )


            ai1, ai2 = st.columns(2)


            with ai1:

                render_ai_card(
                    "Video Summary",
                    analysis.get(
                        "summary",
                        "No summary available.",
                    ),
                )


            with ai2:

                render_ai_card(
                    "Detected Objects",
                    objects_text,
                )


            st.write("")


            ai3, ai4 = st.columns(2)


            with ai3:

                render_ai_card(
                    "Major Actions",
                    actions_text,
                )


            with ai4:

                render_ai_card(
                    "Temporal Sequence",
                    temporal_text,
                )


        else:

            ai1, ai2 = st.columns(2)


            with ai1:

                render_ai_card(
                    "Video Summary",
                    (
                        "Run AI analysis to generate "
                        "a video summary."
                    ),
                    placeholder=True,
                )


            with ai2:

                render_ai_card(
                    "Detected Objects",
                    (
                        "Run AI analysis to identify "
                        "important objects."
                    ),
                    placeholder=True,
                )


            st.write("")


            ai3, ai4 = st.columns(2)


            with ai3:

                render_ai_card(
                    "Major Actions",
                    (
                        "Run AI analysis to identify "
                        "major actions."
                    ),
                    placeholder=True,
                )


            with ai4:

                render_ai_card(
                    "Temporal Sequence",
                    (
                        "Run AI analysis to infer "
                        "chronological events."
                    ),
                    placeholder=True,
                )


        # ====================================================
        # ASK VIDEO
        # ====================================================

        render_section(
            "04 / ASK VIDEO",
            "Ask the Video-LLM",
            (
                "Ask questions about objects, actions, "
                "events, and before-after relationships "
                "in the uploaded video."
            ),
        )


        question = st.text_input(
            "Question",
            placeholder=(
                "What happened after the laptop was opened?"
            ),
            label_visibility="collapsed",
        )


        if st.button(
            "Ask Video-LLM"
        ):

            if not OPENAI_API_KEY:

                st.error(
                    "OPENAI_API_KEY is not configured."
                )

            elif not frames:

                st.warning(
                    "Analyze the video first."
                )

            elif not question.strip():

                st.warning(
                    "Enter a question first."
                )

            else:

                try:

                    with st.spinner(
                        "Reasoning over the video timeline..."
                    ):

                        st.session_state.last_answer = (
                            ask_video_question(
                                frames,
                                question,
                            )
                        )

                except Exception as error:

                    st.error(
                        f"Question answering failed: "
                        f"{error}"
                    )


        if st.session_state.last_answer:

            answer = safe_text(
                st.session_state.last_answer
            )

            st.html(
                f"""
<div class="answer-box">

    <div class="ai-label">
        Video-LLM Response
    </div>

    {answer}

</div>
"""
            )


        st.html(
            """
<div>

    <span class="question-chip">
        What objects appear in the video?
    </span>

    <span class="question-chip">
        What are the main actions?
    </span>

    <span class="question-chip">
        What happened first?
    </span>

    <span class="question-chip">
        What happened before the tablet was placed on the desk?
    </span>

    <span class="question-chip">
        Summarize the video.
    </span>

</div>
"""
        )


        # ====================================================
        # PIPELINE
        # ====================================================

        render_section(
            "SYSTEM ARCHITECTURE",
            "Video Intelligence Pipeline",
            "",
        )


        st.html(
            """
<div class="pipeline">

    <div class="pipeline-node">

        <div class="pipeline-node-title">
            VIDEO INPUT
        </div>

        <div class="pipeline-node-sub">
            Raw video
        </div>

    </div>


    <div class="pipeline-arrow">
        &gt;
    </div>


    <div class="pipeline-node">

        <div class="pipeline-node-title">
            FRAME SAMPLING
        </div>

        <div class="pipeline-node-sub">
            OpenCV
        </div>

    </div>


    <div class="pipeline-arrow">
        &gt;
    </div>


    <div class="pipeline-node">

        <div class="pipeline-node-title">
            MULTIMODAL LLM
        </div>

        <div class="pipeline-node-sub">
            Visual context
        </div>

    </div>


    <div class="pipeline-arrow">
        &gt;
    </div>


    <div class="pipeline-node">

        <div class="pipeline-node-title">
            TEMPORAL REASONING
        </div>

        <div class="pipeline-node-sub">
            Event order
        </div>

    </div>


    <div class="pipeline-arrow">
        &gt;
    </div>


    <div class="pipeline-node">

        <div class="pipeline-node-title">
            SUMMARY + Q&A
        </div>

        <div class="pipeline-node-sub">
            Natural language
        </div>

    </div>

</div>
"""
        )


        # ====================================================
        # FEASIBILITY TEST
        # ====================================================

        render_section(
            "05 / FEASIBILITY TEST",
            "Prototype Evaluation",
            (
                "Run five standardized questions and "
                "manually evaluate each generated answer "
                "against the actual contents of the video."
            ),
        )


        feasibility_questions = [

            (
                "Object Recognition",
                (
                    "What important objects are visible "
                    "in the video?"
                ),
            ),

            (
                "Action Recognition",
                (
                    "What are the major actions performed "
                    "in the video?"
                ),
            ),

            (
                "Temporal Order",
                (
                    "Describe the main actions in "
                    "chronological order."
                ),
            ),

            (
                "Before/After Reasoning",
                (
                    "Identify one meaningful before-and-after "
                    "relationship visible in the video."
                ),
            ),

            (
                "Summarization",
                (
                    "Summarize the entire video "
                    "in one sentence."
                ),
            ),
        ]


        if st.button(
            "Run 5-Question Feasibility Test"
        ):

            if not OPENAI_API_KEY:

                st.error(
                    "OPENAI_API_KEY is not configured."
                )

            elif not frames:

                st.warning(
                    "Analyze the video first."
                )

            else:

                new_answers = {}

                progress = st.progress(0)


                try:

                    for index, (
                        category,
                        test_question,
                    ) in enumerate(
                        feasibility_questions,
                        start=1,
                    ):

                        with st.spinner(
                            f"Running test "
                            f"{index} of 5: "
                            f"{category}"
                        ):

                            answer = (
                                ask_video_question(
                                    frames,
                                    test_question,
                                )
                            )

                            new_answers[
                                category
                            ] = {
                                "question":
                                    test_question,

                                "answer":
                                    answer,
                            }

                        progress.progress(
                            index / 5
                        )


                    st.session_state.feasibility_answers = (
                        new_answers
                    )


                    for key in list(
                        st.session_state.keys()
                    ):
                        if key.startswith(
                            "evaluation_"
                        ):
                            del st.session_state[key]


                    st.success(
                        "Feasibility questions completed. "
                        "Now manually evaluate each response."
                    )


                except Exception as error:

                    st.error(
                        f"Feasibility test failed: "
                        f"{error}"
                    )


        feasibility_answers = (
            st.session_state.feasibility_answers
        )


        if feasibility_answers:

            evaluation_scores = {}

            score_map = {
                "Correct": 1.0,
                "Partially Correct": 0.5,
                "Incorrect": 0.0,
            }


            for index, (
                category,
                data,
            ) in enumerate(
                feasibility_answers.items(),
                start=1,
            ):

                with st.expander(
                    f"{index}. {category}",
                    expanded=True,
                ):

                    st.markdown(
                        f"**Question:** "
                        f"{data['question']}"
                    )

                    st.markdown(
                        "**Video-LLM Answer:**"
                    )

                    st.write(
                        data["answer"]
                    )


                    evaluation = st.radio(
                        "Human Evaluation",
                        [
                            "Correct",
                            "Partially Correct",
                            "Incorrect",
                        ],
                        index=None,
                        key=(
                            f"evaluation_{category}"
                        ),
                        horizontal=True,
                    )


                    if evaluation is not None:

                        evaluation_scores[
                            category
                        ] = score_map[
                            evaluation
                        ]


            # ------------------------------------------------
            # SCORE ONLY AFTER ALL FIVE HAVE BEEN EVALUATED
            # ------------------------------------------------

            if (
                len(evaluation_scores)
                == len(feasibility_answers)
            ):

                total_score = sum(
                    evaluation_scores.values()
                )

                max_score = len(
                    feasibility_answers
                )

                percentage = (
                    total_score
                    / max_score
                    * 100
                )


                render_section(
                    "EVALUATION RESULT",
                    "Feasibility Score",
                    (
                        "Final score based on human "
                        "evaluation of all five responses."
                    ),
                )


                score1, score2, score3 = (
                    st.columns(3)
                )


                with score1:

                    st.html(
                        f"""
<div class="metric-card">

    <div class="metric-label">
        Total Score
    </div>

    <div class="metric-value">
        {total_score:.1f} / {max_score}
    </div>

    <div class="metric-sub">
        Human evaluated
    </div>

</div>
"""
                    )


                with score2:

                    st.html(
                        f"""
<div class="metric-card">

    <div class="metric-label">
        Feasibility
    </div>

    <div class="metric-value">
        {percentage:.0f}%
    </div>

    <div class="metric-sub">
        Prototype result
    </div>

</div>
"""
                    )


                with score3:

                    st.html(
                        f"""
<div class="metric-card">

    <div class="metric-label">
        Model
    </div>

    <div class="metric-value">
        {safe_text(OPENAI_MODEL)}
    </div>

    <div class="metric-sub">
        Multimodal analysis
    </div>

</div>
"""
                    )


                # ============================================
                # DOWNLOADABLE REPORT
                # ============================================

                report_lines = [

                    "Video-LLM Classroom Assistant",

                    (
                        "CVPR 2025 Video-LLM "
                        "Challenge Feasibility Test"
                    ),

                    "",

                    "VIDEO METADATA",

                    (
                        f"Filename: "
                        f"{uploaded_file.name}"
                    ),

                    (
                        f"Duration: "
                        f"{metadata['duration']:.2f} seconds"
                    ),

                    (
                        f"FPS: "
                        f"{metadata['fps']:.2f}"
                    ),

                    (
                        f"Resolution: "
                        f"{metadata['width']} x "
                        f"{metadata['height']}"
                    ),

                    (
                        f"Frames sampled: "
                        f"{len(frames)}"
                    ),

                    (
                        f"Model: "
                        f"{OPENAI_MODEL}"
                    ),

                    "",
                ]


                if analysis:

                    report_lines.extend(
                        [
                            "VIDEO SUMMARY",

                            analysis.get(
                                "summary",
                                "",
                            ),

                            "",

                            "DETECTED OBJECTS",

                            ", ".join(
                                str(item)
                                for item in analysis.get(
                                    "objects",
                                    [],
                                )
                            ),

                            "",

                            "MAJOR ACTIONS",

                            "\n".join(
                                str(item)
                                for item in analysis.get(
                                    "actions",
                                    [],
                                )
                            ),

                            "",

                            "TEMPORAL SEQUENCE",

                            "\n".join(
                                str(item)
                                for item in analysis.get(
                                    "temporal_sequence",
                                    [],
                                )
                            ),

                            "",
                        ]
                    )


                report_lines.extend(
                    [
                        "FEASIBILITY TEST",
                        "",
                    ]
                )


                for category, data in (
                    feasibility_answers.items()
                ):

                    selected_evaluation = (
                        st.session_state.get(
                            f"evaluation_{category}"
                        )
                    )

                    report_lines.extend(
                        [
                            (
                                f"Category: "
                                f"{category}"
                            ),

                            (
                                f"Question: "
                                f"{data['question']}"
                            ),

                            (
                                f"Model Response: "
                                f"{data['answer']}"
                            ),

                            (
                                f"Human Evaluation: "
                                f"{selected_evaluation}"
                            ),

                            (
                                f"Human Score: "
                                f"{evaluation_scores[category]}"
                            ),

                            "",
                        ]
                    )


                report_lines.extend(
                    [
                        (
                            f"FINAL SCORE: "
                            f"{total_score:.1f} / "
                            f"{max_score}"
                        ),

                        (
                            f"FEASIBILITY: "
                            f"{percentage:.0f}%"
                        ),

                        "",

                        "CONCLUSION",

                        (
                            "The prototype demonstrates "
                            "the feasibility of using "
                            "multimodal LLMs with sampled "
                            "video frames for visual "
                            "understanding, temporal "
                            "reasoning, summarization, "
                            "and natural-language "
                            "question answering."
                        ),
                    ]
                )


                report_text = "\n".join(
                    report_lines
                )


                st.download_button(
                    label=(
                        "Download Experiment Results"
                    ),

                    data=report_text,

                    file_name=(
                        "video_llm_"
                        "feasibility_results.txt"
                    ),

                    mime="text/plain",
                )


            else:

                remaining = (
                    len(feasibility_answers)
                    - len(evaluation_scores)
                )

                st.info(
                    f"Evaluate all five responses "
                    f"to calculate the final score. "
                    f"{remaining} evaluation(s) remaining."
                )


else:

    st.html(
        """
<div class="glass-card">

    <div class="ai-label">
        Waiting for Video
    </div>

    <div class="ai-placeholder">
        Upload an MP4, MOV, or AVI file to begin
        the Video-LLM feasibility experiment.
    </div>

</div>
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
<div class="footer">

    Video-LLM Classroom Assistant

    <br>

    Computer Vision Challenge |
    CVPR 2025 Video-LLM Workshop

</div>
"""
)