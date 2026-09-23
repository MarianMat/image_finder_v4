import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.image_service import process_image
from src.vector_store import ImageVectorStore

load_dotenv()

st.set_page_config(
    page_title="AI Image Finder V4",
    page_icon="🖼️",
    layout="wide",
)

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
QDRANT_DIR = DATA_DIR / "qdrant"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
QDRANT_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "image_descriptions")

@st.cache_resource
def get_store():
    return ImageVectorStore(
        path=str(QDRANT_DIR),
        collection_name=COLLECTION_NAME,
    )

store = get_store()

st.title("🖼️ AI Image Finder — V4")
st.caption("Upload images → AI description → embedding → Qdrant → semantic image search")

with st.sidebar:
    st.header("Ustawienia")
    top_k = st.slider("Liczba wyników wyszukiwania", 1, 20, 6)
    min_score = st.slider("Minimalne podobieństwo", 0.0, 1.0, 0.25, 0.05)

    st.divider()
    st.write(f"**Kolekcja Qdrant:** `{COLLECTION_NAME}`")
    st.write(f"**Model vision:** `{os.getenv('OPENAI_VISION_MODEL', 'gpt-4o')}`")
    st.write(
        f"**Embedding:** `{os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large')}`"
    )

tab_upload, tab_search = st.tabs(["📤 Dodaj zdjęcia", "🔎 Wyszukaj zdjęcia"])

with tab_upload:
    st.subheader("Prześlij jedno lub wiele zdjęć")

    uploaded_files = st.file_uploader(
        "Wybierz pliki",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Zdjęcia zostaną zapisane lokalnie, opisane przez model vision i zapisane jako embedding w Qdrant.",
    )

    if uploaded_files:
        st.write(f"Wybrano: **{len(uploaded_files)}** plik(ów)")

        if st.button("🚀 Przetwórz zdjęcia", type="primary"):
            if not os.getenv("OPENAI_API_KEY"):
                st.error("Brak OPENAI_API_KEY. Uzupełnij plik .env.")
                st.stop()

            progress = st.progress(0)
            status = st.empty()

            success_count = 0
            error_count = 0

            for index, uploaded_file in enumerate(uploaded_files, start=1):
                status.info(f"Przetwarzanie {index}/{len(uploaded_files)}: {uploaded_file.name}")

                try:
                    result = process_image(
                        uploaded_file=uploaded_file,
                        images_dir=IMAGES_DIR,
                        store=store,
                    )

                    success_count += 1

                    with st.expander(f"✅ {result['filename']}", expanded=False):
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            st.image(result["image_path"], use_container_width=True)

                        with col2:
                            st.write("**Opis wygenerowany przez AI:**")
                            st.write(result["description"])
                            st.caption(f"ID: {result['image_id']}")

                except Exception as exc:
                    error_count += 1
                    st.error(f"❌ {uploaded_file.name}: {exc}")

                progress.progress(index / len(uploaded_files))

            status.success(
                f"Gotowe. Przetworzono: {success_count}, błędy: {error_count}."
            )

with tab_search:
    st.subheader("Wyszukiwanie semantyczne")

    query = st.text_input(
        "Opisz, czego szukasz",
        placeholder="np. samochód stojący na parkingu obok budynku",
    )

    if st.button("🔎 Szukaj", type="primary", disabled=not query.strip()):
        if not os.getenv("OPENAI_API_KEY"):
            st.error("Brak OPENAI_API_KEY. Uzupełnij plik .env.")
            st.stop()

        with st.spinner("Konwertuję zapytanie na embedding i przeszukuję Qdrant..."):
            results = store.search(
                query=query.strip(),
                limit=top_k,
                score_threshold=min_score,
            )

        if not results:
            st.warning("Nie znaleziono zdjęć spełniających ustawiony próg podobieństwa.")
        else:
            st.success(f"Znaleziono {len(results)} wyników.")

            columns = st.columns(3)

            for index, result in enumerate(results):
                with columns[index % 3]:
                    st.image(
                        result["image_path"],
                        use_container_width=True,
                    )
                    st.markdown(f"**{result['filename']}**")
                    st.caption(f"Similarity: {result['score']:.3f}")
                    st.write(result["description"])

                    with st.expander("Metadata"):
                        st.json({
                            "image_id": result["image_id"],
                            "filename": result["filename"],
                            "score": result["score"],
                            "created_at": result["created_at"],
                        })
