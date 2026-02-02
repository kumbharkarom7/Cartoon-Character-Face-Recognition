import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# =========================
# Load trained model
# =========================
model = tf.keras.models.load_model("model.h5")

# =========================
# Class names (ORDER MUST MATCH TRAINING)
# =========================
class_names = [
    "bugsbunny",
    "donald duck",
    "doraemon",
    "garfield",
    "jerry",
    "mickey",
    "minion",
    "pikachu",
    "popeye",
    "scoobydoo",
    "shinchan",
    "spongebob",
    "tom"
]

# =========================
# Character Information
# =========================
CHARACTER_INFO = {
    "bugsbunny": {
        "name": "Bugs Bunny",
        "cartoon": "Looney Tunes",
        "creator": "Tex Avery",
        "first_appearance": "1938",
        "personality": "Clever, sarcastic, confident",
        "special_ability": "Outsmarts enemies",
        "fun_fact": "Famous line: What's up, Doc?"
    },
    "donald duck": {
        "name": "Donald Duck",
        "cartoon": "Disney",
        "creator": "Walt Disney",
        "first_appearance": "1934",
        "personality": "Short-tempered, funny",
        "special_ability": "Comic anger",
        "fun_fact": "Wears sailor outfit"
    },
    "doraemon": {
        "name": "Doraemon",
        "cartoon": "Doraemon",
        "creator": "Fujiko F. Fujio",
        "first_appearance": "1969",
        "personality": "Helpful, kind",
        "special_ability": "Future gadgets",
        "fun_fact": "Afraid of mice"
    },
    "garfield": {
        "name": "Garfield",
        "cartoon": "Garfield",
        "creator": "Jim Davis",
        "first_appearance": "1978",
        "personality": "Lazy, sarcastic",
        "special_ability": "Extreme laziness",
        "fun_fact": "Loves lasagna"
    },
    "jerry": {
        "name": "Jerry",
        "cartoon": "Tom and Jerry",
        "creator": "Hanna-Barbera",
        "first_appearance": "1940",
        "personality": "Smart, playful",
        "special_ability": "Outsmarts Tom",
        "fun_fact": "Rarely speaks"
    },
    "mickey": {
        "name": "Mickey Mouse",
        "cartoon": "Disney",
        "creator": "Walt Disney",
        "first_appearance": "1928",
        "personality": "Cheerful, kind",
        "special_ability": "Leadership",
        "fun_fact": "Disney mascot"
    },
    "minion": {
        "name": "Minion",
        "cartoon": "Despicable Me",
        "creator": "Illumination",
        "first_appearance": "2010",
        "personality": "Funny, childish",
        "special_ability": "Comedy",
        "fun_fact": "Speaks Minionese"
    },
    "pikachu": {
        "name": "Pikachu",
        "cartoon": "Pokémon",
        "creator": "Satoshi Tajiri",
        "first_appearance": "1996",
        "personality": "Energetic, loyal",
        "special_ability": "Electric shocks",
        "fun_fact": "Pokémon mascot"
    },
    "popeye": {
        "name": "Popeye",
        "cartoon": "Popeye the Sailor",
        "creator": "E. C. Segar",
        "first_appearance": "1929",
        "personality": "Brave",
        "special_ability": "Strength from spinach",
        "fun_fact": "Promoted spinach"
    },
    "scoobydoo": {
        "name": "Scooby-Doo",
        "cartoon": "Scooby-Doo",
        "creator": "Hanna-Barbera",
        "first_appearance": "1969",
        "personality": "Cowardly but lovable",
        "special_ability": "Solves mysteries",
        "fun_fact": "Loves Scooby Snacks"
    },
    "shinchan": {
        "name": "Shinchan",
        "cartoon": "Crayon Shin-chan",
        "creator": "Yoshito Usui",
        "first_appearance": "1990",
        "personality": "Naughty, funny",
        "special_ability": "Unpredictable humor",
        "fun_fact": "Famous dance"
    },
    "spongebob": {
        "name": "SpongeBob SquarePants",
        "cartoon": "SpongeBob SquarePants",
        "creator": "Stephen Hillenburg",
        "first_appearance": "1999",
        "personality": "Happy, optimistic",
        "special_ability": "Endless positivity",
        "fun_fact": "Lives in pineapple"
    },
    "tom": {
        "name": "Tom",
        "cartoon": "Tom and Jerry",
        "creator": "Hanna-Barbera",
        "first_appearance": "1940",
        "personality": "Persistent",
        "special_ability": "Never gives up",
        "fun_fact": "Almost never wins"
    }
}

# =========================
# Streamlit UI
# =========================
st.title("🎨 Cartoon Character Face Recognition")
st.write("Upload a cartoon face image to identify the character")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=500)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions)
    confidence = predictions[0][pred_index]

    predicted_class = class_names[pred_index]
    info = CHARACTER_INFO[predicted_class]

    # Display result
    st.markdown("## 🎯 Prediction Result")
    st.write(f"**Character Name:** {info['name']}")
    st.write(f"**Cartoon Series:** {info['cartoon']}")
    st.write(f"**Creator:** {info['creator']}")
    st.write(f"**First Appearance:** {info['first_appearance']}")
    st.write(f"**Personality:** {info['personality']}")
    st.write(f"**Special Ability:** {info['special_ability']}")
    st.write(f"**Fun Fact:** {info['fun_fact']}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")
