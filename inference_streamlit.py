import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse

import streamlit as st
import torch
from torch import nn

import utils.data_utils as data_utils
import utils.model_utils as model_utils
from models.transformer.encoder import Encoder
from models.transformer.decoder import Decoder
from models.transformer.seq2seq import Seq2Seq


def load_tokenizers_and_model(config_fpath):
    config = data_utils.get_config(config_fpath)
    for key, value in config.items():
        globals()[key] = value

    src_tok_fpath = "/".join([checkpoint["dir"], checkpoint["tokenizer"]["src"]])
    tgt_tok_fpath = "/".join([checkpoint["dir"], checkpoint["tokenizer"]["tgt"]])
    src_tok = torch.load(src_tok_fpath)
    tgt_tok = torch.load(tgt_tok_fpath)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc_voc_size = len(src_tok.vocab)
    dec_voc_size = len(tgt_tok.vocab)

    enc = Encoder(input_dim=enc_voc_size,
                    hid_dim=d_model,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    pf_dim=ffn_hidden,
                    dropout=drop_prob,
                    device=device,
                    max_length=max_len)

    dec = Decoder(output_dim=dec_voc_size,
                    hid_dim=d_model,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    pf_dim=ffn_hidden,
                    dropout=drop_prob,
                    device=device,
                    max_length=max_len)

    src_pad_id = src_tok.vocab.pad_id
    tgt_pad_id = tgt_tok.vocab.pad_id

    model = Seq2Seq(enc, dec, src_pad_id, tgt_pad_id, device).to(device)

    best_checkpoint_fpath = "/".join([checkpoint["dir"], checkpoint["best"]])
    checkpoint_dict = torch.load(best_checkpoint_fpath)
    model.load_state_dict(checkpoint_dict["model_state_dict"])

    return src_tok, tgt_tok, model

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1452868195396-89c1af3b1b2e");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def main(config_fpath="config.yml"):
    config = data_utils.get_config(config_fpath)
    for key, value in config.items():
        globals()[key] = value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_tok, tgt_tok, model = load_tokenizers_and_model(config_fpath)


    st.set_page_config(page_title="En-Vi Neural Machine Translation", page_icon="👨‍💻")
    add_bg_from_url() 
    
    st.title("En-Vi Neural Machine Translation")

    # Define a two-column layout
    left_column, right_column = st.columns(2)

    # Left column for input text
    with left_column:
        input_text = st.text_area("English Input Text", "Hello, how are you?", height=200,
                                  max_chars=400)

    # Right column for translated text
    with right_column:
        output_text = st.empty()
        output_text.text_area("Vietnamese Output Text", height=200)

    if st.button("Translate"):
        pred_tokens, attention = model_utils.translate_sentence(input_text, 
                                                                src_tok, tgt_tok, 
                                                                model, device, 
                                                                max_len)

        translated_text = " ".join(pred_tokens[1:-1])
        output_text.text_area("Vietnamese Output Text", translated_text, height=200)


        st.header("Attention Matrix")
        src_tokens = [token.lower() for token in src_tok.tokenize(input_text)] 
        src_tokens = [src_tok.vocab.bos_token] + src_tokens + [src_tok.vocab.eos_token]
        fig = model_utils.display_attention(src_tokens, pred_tokens[1:], 
                                            attention, n_heads=1, 
                                            n_rows=1, n_cols=1, fig_size=(5, 5))
        st.pyplot(fig)

        st.header("Beam Search Results")
        candidates = model_utils.translate_sentence_beam_search(input_text,
                                                                src_tok, tgt_tok, 
                                                                model, device, 
                                                                max_len, beam_size)
        # cut off <bos> and <eos> tokens
        candidates = [(tokens[1:-1], score) for tokens, score in candidates]
        text = ""
        for i, (tokens, log) in enumerate(candidates):
            text += f"Translation {i+1}:  Log value = {log:.2f} -  {' '.join(tokens)}\n"
        st.text_area("Beam Search Results", text, height=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Host web app with streamlit")

    parser.add_argument("--config",
                        default="config.yml",
                        help="path to config file",
                        dest="config_fpath")

    args = parser.parse_args()

    main(**vars(args))
