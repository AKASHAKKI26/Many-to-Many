# Many-to-Many RNN
# Named Entity Recognition using SimpleRNN

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Embedding, Input, SimpleRNN, TimeDistributed
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# File Names
MODEL = "ner_model.keras"
TOKENIZER = "tokenizer.pkl"
TAG2INDEX = "tag2index.pkl"
INDEX2TAG = "index2tag.pkl"
MAX_LEN = 50

st.set_page_config(
    page_title="Named Entity Recognition",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        }

        .hero {
            padding: 2rem 2.2rem;
            border-radius: 18px;
            background: linear-gradient(120deg, #6366f1 0%, #8b5cf6 45%, #ec4899 100%);
            box-shadow: 0 10px 30px rgba(99, 102, 241, 0.35);
            margin-bottom: 1.6rem;
        }
        .hero h1 {
            color: #ffffff;
            font-size: 2.1rem;
            margin-bottom: 0.2rem;
        }
        .hero p {
            color: rgba(255,255,255,0.9);
            font-size: 1.02rem;
            margin: 0;
        }

        .card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.2rem;
        }

        .entity-chip {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            margin: 0.2rem 0.25rem;
            border-radius: 10px;
            font-size: 0.95rem;
            font-weight: 500;
            line-height: 1.6;
        }
        .entity-tag {
            font-size: 0.65rem;
            font-weight: 700;
            opacity: 0.85;
            margin-left: 0.35rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        .legend-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.6rem;
            margin: 0.15rem;
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 600;
        }

        div[data-testid="stTextInput"] input {
            border-radius: 10px;
            padding: 0.6rem 0.8rem;
        }

        div[data-testid="stButton"] button {
            border-radius: 10px;
            font-weight: 600;
            padding: 0.5rem 1.4rem;
            background: linear-gradient(120deg, #6366f1, #8b5cf6);
            color: white;
            border: none;
        }
        div[data-testid="stButton"] button:hover {
            background: linear-gradient(120deg, #4f46e5, #7c3aed);
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Data Loading + Preprocessing + Training
# (Only ever runs once, thanks to caching.
#  Streamlit reruns the whole script on every
#  click, so anything expensive must be cached
#  or it repeats every single interaction.)
@st.cache_resource(show_spinner="Loading model (training on first run)...")
def train_or_load_model():

    # If a trained model already exists, skip straight to loading —
    # no need to touch the CSV or retrain at all.
    if os.path.exists(MODEL):

        model = load_model(MODEL)

        with open(TOKENIZER, "rb") as f:
            tokenizer = pickle.load(f)

        with open(TAG2INDEX, "rb") as f:
            tag2index = pickle.load(f)

        with open(INDEX2TAG, "rb") as f:
            index2tag = pickle.load(f)

        return model, tokenizer, tag2index, index2tag

    # ---------- Read Dataset ----------
    # Different copies of this dataset ship with different encodings, so
    # try a few in order rather than hardcoding one.
    df = None
    last_error = None
    for enc in ("utf-8-sig", "latin1", "utf-8"):
        try:
            import zipfile

            # Extract ner.csv from ner.zip if it doesn't already exist
            if not os.path.exists("ner.csv"):
                with zipfile.ZipFile("ner.zip", "r") as zip_ref:
                    zip_ref.extractall()

            df = pd.read_csv(
                    "ner.csv" ,
                    engine="python",
                    on_bad_lines="skip",
                    delimiter=",",
                    encoding=enc,
                    keep_default_na=False,
                )
            break
        except Exception as e:
            last_error = e

    if df is None:
        raise RuntimeError(f"Could not read ner.csv with any known encoding: {last_error}")

    # Normalize column names: strip whitespace/BOM artifacts so lookups
    # don't fail due to hidden formatting differences.
    df.columns = [str(c).strip() for c in df.columns]

    # Build a case-insensitive lookup so we handle schema variants like
    # 'Word'/'word', 'Tag'/'tag', 'Sentence #'/'sentence_idx', etc.
    lower_to_actual = {c.lower(): c for c in df.columns}

    sentence_col_candidates = [
        "sentence #", "sentence#", "sentence_#", "sentence",
        "sentence_idx", "sentence_id",
    ]
    word_col_candidates = ["word"]
    tag_col_candidates = ["tag"]

    def find_col(candidates):
        for cand in candidates:
            if cand in lower_to_actual:
                return lower_to_actual[cand]
        return None

    sentence_col = find_col(sentence_col_candidates)
    word_col = find_col(word_col_candidates)
    tag_col = find_col(tag_col_candidates)

    if sentence_col is None or word_col is None or tag_col is None:
        raise KeyError(
            "Expected columns like 'Sentence #'/'sentence_idx', 'Word', 'Tag' were not found. "
            f"Columns actually present in ner.csv: {list(df.columns)}"
        )

    # Standardize to the names the rest of the pipeline expects.
    df = df.rename(columns={sentence_col: "Sentence #", word_col: "Word", tag_col: "Tag"})

    # Keep only the columns we actually need — this dataset variant has
    # many extra engineered features (lemma, shape, prev-word, etc.)
    # that aren't relevant to this model.
    df = df[["Sentence #", "Word", "Tag"]].copy()

    # Some rows can end up with missing Word/Tag values (e.g. malformed
    # lines in the source CSV). Drop those and force everything to plain
    # strings so " ".join(...) never chokes on a None/NaN entry.
    df = df.dropna(subset=["Word", "Tag"])
    df["Word"] = df["Word"].astype(str)
    df["Tag"] = df["Tag"].astype(str)
    df = df[(df["Word"].str.strip() != "") & (df["Tag"].str.strip() != "")]

    # ---------- Fill Missing Sentence Numbers ----------
    df["Sentence #"] = df["Sentence #"].replace("", np.nan).ffill()

    # ---------- Group Words by Sentence ----------
    grouped = df.groupby("Sentence #")

    sentences = []
    tags = []

    for sentence_no, data in grouped:
        words = list(data["Word"].values)
        ner_tags = list(data["Tag"].values)
        sentences.append(words)
        tags.append(ner_tags)

    # Convert list of words into a single string sentence
    joined_sentences = [" ".join(sentence) for sentence in sentences]

    # ---------- Tokenizer ----------
    tokenizer = Tokenizer(lower=False, oov_token="<OOV>")
    tokenizer.fit_on_texts(joined_sentences)

    vocab_size = len(tokenizer.word_index) + 1

    # ---------- Convert Sentences to Sequences ----------
    X = tokenizer.texts_to_sequences(joined_sentences)
    X = pad_sequences(X, maxlen=MAX_LEN, padding="post")

    # ---------- Tag Dictionaries ----------
    unique_tags = sorted(set(df["Tag"].unique()))

    tag2index = {tag: i for i, tag in enumerate(unique_tags)}
    index2tag = {i: tag for tag, i in tag2index.items()}

    # ---------- Encode Output Tags ----------
    y = [[tag2index[tag] for tag in sentence_tags] for sentence_tags in tags]
    y = pad_sequences(y, maxlen=MAX_LEN, padding="post", value=tag2index["O"])

    num_tags = len(tag2index)

    # ---------- Train Test Split ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------- Save Tokenizer + Dictionaries ----------
    with open(TOKENIZER, "wb") as f:
        pickle.dump(tokenizer, f)

    with open(TAG2INDEX, "wb") as f:
        pickle.dump(tag2index, f)

    with open(INDEX2TAG, "wb") as f:
        pickle.dump(index2tag, f)

    # ---------- Convert Output to 3D for sparse_categorical_crossentropy ----------
    y_train = np.expand_dims(y_train, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)

    # ---------- Build Model ----------
    model = Sequential()
    model.add(Input(shape=(MAX_LEN,)))
    model.add(Embedding(input_dim=vocab_size, output_dim=64))
    model.add(SimpleRNN(128, return_sequences=True))
    model.add(TimeDistributed(Dense(64, activation="relu")))
    model.add(TimeDistributed(Dense(num_tags, activation="softmax")))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    # ---------- Train ----------
    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early],
    )

    model.save(MODEL)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.write(f"Training complete — Test Accuracy: {accuracy * 100:.2f}%")

    return model, tokenizer, tag2index, index2tag

# Tag Colors + Rendering Helpers
TAG_COLORS = {
    "per": ("#fecaca", "#7f1d1d"),   # person
    "org": ("#bfdbfe", "#1e3a8a"),   # organization
    "geo": ("#bbf7d0", "#14532d"),   # geographical location
    "gpe": ("#bbf7d0", "#14532d"),   # geopolitical entity
    "loc": ("#bbf7d0", "#14532d"),   # location
    "tim": ("#fde68a", "#78350f"),   # time
    "art": ("#e9d5ff", "#581c87"),   # artifact
    "eve": ("#fbcfe8", "#831843"),   # event
    "nat": ("#a5f3fc", "#164e63"),   # nationality/religious/political
}
DEFAULT_CHIP = ("rgba(255,255,255,0.08)", "#e5e7eb")  # for "O" / unrecognized


def tag_family(tag: str) -> str:
    """Extract the entity family from a tag like 'B-per' or 'I-geo'."""
    if "-" in tag:
        return tag.split("-", 1)[1].lower()
    return tag.lower()


def chip_colors(tag: str):
    family = tag_family(tag)
    return TAG_COLORS.get(family, DEFAULT_CHIP)


def render_highlighted_sentence(word_tag_pairs) -> str:
    chips = []
    for word, tag in word_tag_pairs:
        bg, fg = chip_colors(tag)
        if tag == "O":
            chips.append(
                f'<span class="entity-chip" style="background:{bg};color:{fg};">{word}</span>'
            )
        else:
            chips.append(
                f'<span class="entity-chip" style="background:{bg};color:{fg};">'
                f"{word}<span class='entity-tag'>{tag}</span></span>"
            )
    return " ".join(chips)


def render_legend():
    labels = {
        "per": "Person", "org": "Organization", "geo": "Geo-Location",
        "gpe": "Geopolitical", "loc": "Location", "tim": "Time",
        "art": "Artifact", "eve": "Event", "nat": "Nationality/Religion",
    }
    chips = []
    for family, label in labels.items():
        bg, fg = TAG_COLORS[family]
        chips.append(
            f'<span class="legend-chip" style="background:{bg};color:{fg};">{label}</span>'
        )
    bg, fg = DEFAULT_CHIP
    chips.append(f'<span class="legend-chip" style="background:{bg};color:{fg};">O — Not an entity</span>')
    return " ".join(chips)


def predict_ner(sentence, model, tokenizer, index2tag):

    words = sentence.split()

    sequence = tokenizer.texts_to_sequences([sentence])
    sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding="post")

    prediction = model.predict(sequence, verbose=0)
    prediction = np.argmax(prediction, axis=-1)

    result = []
    for i in range(min(len(words), MAX_LEN)):
        tag = index2tag[prediction[0][i]]
        result.append((words[i], tag))

    return result


# Streamlit User Interface
st.markdown(
    """
    <div class="hero">
        <h1>Named Entity Recognition</h1>
        <p>Many-to-Many RNN (SimpleRNN) · tag every word in a sentence as a person, place, org, date, and more</p>
    </div>
    """,
    unsafe_allow_html=True,
)

EXAMPLES = [
    "Barack Obama visited Germany last Friday",
    "Apple announced a new iPhone in California",
    "The United Nations held a meeting in New York",
    "Elon Musk founded SpaceX in the United States",
]

with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This app tags every word in a sentence with its entity type "
        "(person, location, organization, date, etc.) using a "
        "SimpleRNN sequence model trained on the GMB NER dataset."
    )
    st.divider()
    st.subheader("✨ Try an example")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["sentence_input"] = ex
    st.divider()
    st.subheader("🎨 Legend")
    st.markdown(render_legend(), unsafe_allow_html=True)

if not os.path.exists("ner.csv") and not os.path.exists("ner.zip") and not os.path.exists(MODEL):
    st.error(
        "No 'ner.csv' found and no saved model exists yet. "
        "Please place 'ner.csv' in the app's working directory so the model can be trained."
    )
else:
    try:
        model, tokenizer, tag2index, index2tag = train_or_load_model()
    except Exception as e:
        st.error(f"Failed to load or train the model: {e}")
        st.stop()

    st.markdown('<div class="card">', unsafe_allow_html=True)

    sentence = st.text_input(
        "Enter a sentence",
        key="sentence_input",
        placeholder="e.g. Barack Obama visited Germany last Friday",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        predict_clicked = st.button("🔍 Predict")

    st.markdown("</div>", unsafe_allow_html=True)

    if predict_clicked:

        if sentence.strip() == "":
            st.warning("Please enter a sentence.")
        else:
            with st.spinner("Tagging entities..."):
                output = predict_ner(sentence, model, tokenizer, index2tag)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Result")
            st.markdown(render_highlighted_sentence(output), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            entities_found = [(w, t) for w, t in output if t != "O"]

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Words analyzed", len(output))
            with col_b:
                st.metric("Entities detected", len(entities_found))

            with st.expander("📋 View detailed word-by-word tags"):
                result_df = pd.DataFrame(output, columns=["Word", "NER Tag"])
                st.dataframe(result_df, use_container_width=True, hide_index=True)