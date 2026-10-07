from __future__ import annotations

import streamlit as st

from src.inference.translator import Translator
from src.model.loader import load_translation_components
from src.utils.config import load_config


st.set_page_config(
    page_title="Neural Machine Translator",
    page_icon="🌍",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading translation model...")
def get_translator(config_path: str = "configs/config.yaml") -> Translator:
    config = load_config(config_path)
    model, tokenizer = load_translation_components(config)
    return Translator(model=model, tokenizer=tokenizer, config=config)


def main() -> None:
    config = load_config("configs/config.yaml")
    app_config = config["app"]
    languages = config["languages"]

    st.title(app_config["title"])
    st.caption(app_config["description"])

    if not languages:
        st.error("No languages are configured in configs/config.yaml.")
        st.stop()

    language_names = list(languages.keys())

    col1, col2 = st.columns(2)

    with col1:
        source_language = st.selectbox(
            "Source language",
            options=language_names,
            index=0,
        )

    with col2:
        default_target_index = 1 if len(language_names) > 1 else 0
        target_language = st.selectbox(
            "Target language",
            options=language_names,
            index=default_target_index,
        )

    source_text = st.text_area(
        "Text to translate",
        height=180,
        placeholder="Enter a sentence...",
    )

    translate_button = st.button(
        "Translate",
        type="primary",
        use_container_width=True,
    )

    if translate_button:
        if not source_text.strip():
            st.warning("Please enter some text.")
            return

        if source_language == target_language:
            st.info("Source and target languages are identical.")
            st.text_area(
                "Translation",
                value=source_text,
                height=180,
                disabled=True,
            )
            return

        try:
            translator = get_translator()

            with st.spinner("Translating..."):
                translation = translator.translate(
                    text=source_text,
                    source_language=source_language,
                    target_language=target_language,
                )

            st.text_area(
                "Translation",
                value=translation,
                height=180,
                disabled=True,
            )

        except Exception as exc:
            st.error(f"Translation failed: {exc}")

    with st.expander("Model information"):
        model_cfg = config["model"]
        st.write(f"**Model source:** `{model_cfg['source']}`")
        st.write(f"**Model path / repository:** `{model_cfg['path_or_repo']}`")
        st.write(f"**INT8 dynamic quantization:** `{model_cfg['quantize_int8']}`")
        st.write(f"**Device:** CPU")


if __name__ == "__main__":
    main()
