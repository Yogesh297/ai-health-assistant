from flask import Flask, request, jsonify
from flask_cors import CORS
import json, os
from ctransformers import AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator
import pandas as pd
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
from huggingface_hub import hf_hub_download

app = Flask(__name__)
CORS(app)

# === Load LLM ===
# Replace with your actual repo ID if different
GGUF_PATH = hf_hub_download(
    repo_id="yyc297/tinyllama-health-model",
    filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    local_dir="models",
    local_dir_use_symlinks=False
)

model = AutoModelForCausalLM.from_pretrained(
    GGUF_PATH,
    model_type="llama",
    gpu_layers=0,
    max_new_tokens=1024,
    temperature=0.6
)

# === FAQ Setup ===
faq_encoder = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv("medical_faqs.csv")
faq_embeddings = faq_encoder.encode(df['question'].tolist())
faq_questions = df['question'].tolist()
faq_answers = df['answer'].tolist()

translator = Translator()

SYSTEM_PROMPT_EN = "You're a helpful health assistant who provides short and friendly replies (2–3 sentences max). You never diagnose, but suggest possibilities and always add: 'This is AI-generated, not a medical diagnosis.'"
SYSTEM_PROMPT_HI = "तुम एक सहायक स्वास्थ्य सहायक हो जो 2-3 पंक्तियों में छोटा और स्पष्ट उत्तर देता है। तुम कभी भी डायग्नोसिस नहीं करते, सिर्फ संभावनाएं बताते हो और हमेशा यह कहते हो: 'यह AI द्वारा जनरेट किया गया उत्तर है, यह चिकित्सा निदान नहीं है।'"

# === Utility Functions for LLM ===
def generate_ai_response(prompt):
    return model(prompt, max_new_tokens=300, temperature=0.3)

def get_faq_response(query):
    query_embed = faq_encoder.encode([query])
    scores = cosine_similarity(query_embed, faq_embeddings)[0]
    max_idx = np.argmax(scores)
    return faq_answers[max_idx] if scores[max_idx] > 0.7 else None

# === Chat History Persistence ===
CHAT_DIR = "chat_history"
os.makedirs(CHAT_DIR, exist_ok=True)

def save_user_chat(user_id, user_input, ai_output):
    filepath = os.path.join(CHAT_DIR, f"{user_id}.json")
    messages = []
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            messages = json.load(f)
    messages.append({"sender": "user", "text": user_input})
    messages.append({"sender": "ai", "text": ai_output})
    with open(filepath, "w") as f:
        json.dump(messages, f, indent=2)

def load_user_chat(user_id):
    filepath = os.path.join(CHAT_DIR, f"{user_id}.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return []

# === API: Message ===
@app.route('/api/message', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')
    lang_hint = data.get('lang', 'en-IN')
    user_id = data.get('userId', 'default')

    if not user_input:
        return jsonify({'response': 'No message provided.'}), 400

    lang = 'hi' if 'hi' in lang_hint else 'en'
    input_en = translator.translate(user_input, src=lang, dest='en').text if lang == 'hi' else user_input

    faq = get_faq_response(input_en)
    if faq:
        final_faq = translator.translate(faq, src='en', dest='hi').text if lang == 'hi' else faq
        save_to_log(user_id, user_input, final_faq)
        return jsonify({'response': final_faq})

    prompt = f"<|system|>{SYSTEM_PROMPT_HI if lang == 'hi' else SYSTEM_PROMPT_EN}<|end|>\n<|user|>{input_en}<|end|>\n<|assistant|>"

    try:
        output = generate_ai_response(prompt).strip()
        final_output = translator.translate(output, src='en', dest='hi').text if lang == 'hi' else output
        save_to_log(user_id, user_input, final_output)
        return jsonify({'response': final_output})
    except Exception:
        return jsonify({'response': "AI is having trouble. Please try again later."})

def save_to_log(user_id, user_input, ai_reply):
    os.makedirs("chat_logs", exist_ok=True)
    path = f"chat_logs/{user_id}.json"
    entry = {
        "text": user_input,
        "sender": "user",
        "timestamp": pd.Timestamp.now().isoformat()
    }
    response = {
        "text": ai_reply,
        "sender": "ai",
        "timestamp": pd.Timestamp.now().isoformat()
    }

    history = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            history = json.load(f)

    history.append(entry)
    history.append(response)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

@app.route('/api/chat', methods=['GET'])
def get_chat_history():
    user_id = request.args.get('userId', 'default')
    path = f"chat_logs/{user_id}.json"
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return jsonify({'messages': json.load(f)})
    return jsonify({'messages': []})
# === IMAGE DIAGNOSIS ===
NUM_CLASSES = 7
model_resnet = models.resnet18(pretrained=False)
model_resnet.fc = torch.nn.Linear(model_resnet.fc.in_features, NUM_CLASSES)
RESNET_PATH = hf_hub_download(
    repo_id="yyc297/tinyllama-health-model",
    filename="resnet18_skin_disease.pt",
    local_dir="models",
    local_dir_use_symlinks=False
)
model_resnet.load_state_dict(torch.load(RESNET_PATH, map_location=torch.device('cpu')))
model_resnet.eval()

skin_labels = {
    0: (
        "Actinic keratoses and intraepithelial carcinoma (AKIEC)",
        "A rough, scaly patch on the skin caused by sun damage. Often found on face, lips, ears, back of hands. "
        "Causes: Chronic sun exposure. Precautions: Avoid sun, use sunscreen, and consult a dermatologist."
    ),
    1: (
        "Basal Cell Carcinoma (BCC)",
        "Common, slow-growing skin cancer. Appears as pearly bump or flat area. "
        "Causes: Long-term sun exposure. Precautions: Use sunscreen, regular checkups."
    ),
    2: (
        "Benign Keratosis-like Lesions (BKL)",
        "Non-cancerous waxy or rough lesions. Causes: Age, sun. Precautions: Monitor changes; usually harmless."
    ),
    3: (
        "Dermatofibroma (DF)",
        "Firm, small skin nodule. Often caused by insect bites or trauma. Precautions: Harmless, consult if growing."
    ),
    4: (
        "Melanoma (MEL)",
        "Deadly skin cancer from moles or dark spots. Causes: UV rays, genetics. Precautions: Watch for ABCDE signs."
    ),
    5: (
        "Melanocytic Nevi (NV)",
        "Common moles. Uniform in shape and color. Precautions: Watch for color, size, border changes."
    ),
    6: (
        "Vascular Lesions (VASC)",
        "Blood vessel clusters under skin. Often genetic or age-related. Precautions: Check if bleeding or changing."
    )
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route('/api/image', methods=['POST'])
def diagnose_image():
    if 'image' not in request.files:
        return jsonify({'response': 'No image uploaded'}), 400

    image = Image.open(request.files['image']).convert('RGB')
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model_resnet(tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(probs).item()
        pred = torch.argmax(probs).item()

    label, desc = skin_labels.get(pred, ("Unknown", "No description available."))
    response = f"{desc} This is an AI-generated guess. Please consult a doctor."

    return jsonify({
        'response': response,
        'diagnosis': {
            'label': label,
            'confidence': round(confidence * 100, 2)
        }
    })

# === Run ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)