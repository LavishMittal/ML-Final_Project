# app.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----- CONFIG -----
MODEL_NAME = "gpt2"  # replace with your fine-tuned model path if needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 100

# ----- Load Model -----
@st.cache_resource(show_spinner=True)
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model(MODEL_NAME)

# ----- Streamlit UI -----
st.title("🛡 Quest Generator Demo")
st.write("Generate quests of different levels, tones, and lengths using your trained model.")

prompt_input = st.text_area("Enter your quest prompt:", "Create a level-1 quest in a forest")

level = st.selectbox("Select Level:", [1, 2, 3, 4, 5])
tone = st.selectbox("Select Tone:", ["dark", "epic", "humorous", "mysterious", "serious"])
length = st.selectbox("Select Length:", ["short", "medium", "long"])

if st.button("Generate Quest"):
    # Build full prompt for the model
    full_prompt = f"[Quest Level {level}, Tone: {tone}, Length: {length}] {prompt_input}"
    
    # Tokenize and generate
    inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    st.subheader("Generated Quest:")
    st.write(generated_text)
