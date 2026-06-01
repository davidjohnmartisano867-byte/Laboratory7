import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. I-load ang pinakamagaling na model (MobileNetV2)
# Ang cache_resource ay nag-i-save sa memory para hindi na niya ulit maglo-load kada mag-upload ng pic
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('MobileNetV2_best_model.h5')
    return model

model = load_model()

# 2. I-configure ang app
st.title("AI 7.0: Cat vs Dog Classifier")
st.write("Nagamit ang **MobileNetV2** bilang pinakamagandang model mula sa aming Transfer Learning Comparison.")
st.write("Mag-upload ng larawan ng pusa o aso para subukan.")

# 3. Ilagay ang class names (Dapat pareho ng order noong training)
# Noong nag-training tayo, 'cat' ang unang naload na folder kaya siya index 0
class_names = ['cat', 'dog']

# 4. File Uploader
uploaded_file = st.file_uploader("Pumili ng larawan...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
        # I-display ang na-upload na image
    img = Image.open(uploaded_file)
    st.image(img, caption="Na-upload na larawan", use_column_width=True)
    
    # 5. I-preprocess ang image para mag-match sa training
    # MAGICAL FIX: I-convert sa RGB para maging 3 channels at mawala ang Alpha/Transparency
    img = img.convert('RGB') 
    # Resize to 224x224
    img = img.resize((224, 224))
    # I-convert to numpy array at i-scale sa [0, 1] (tulad ng ginawa natin sa Colab)
    img_array = np.array(img, dtype=np.float32) / 255.0
    # Mag-add ng batch dimension (kailangan ng model na may shape na [1, 224, 224, 3])
    img_array = np.expand_dims(img_array, axis=0)
    
    # 6. Mag-predict
    with st.spinner('Nag-iisip ang AI...'):
        predictions = model.predict(img_array)
        score = predictions[0][0] # Kunin ang score (0.0 to 1.0)
    
    # 7. I-interpret ang result
    # Dahil sigmoid ang gamit, mas mababa sa 0.5 ay 'cat', mas mataas ay 'dog'
    if score < 0.5:
        predicted_class = class_names[0]
        confidence = (1 - score) * 100
    else:
        predicted_class = class_names[1]
        confidence = score * 100
        
    # 8. I-display ang final result
    st.write("---")
    st.subheader("Resulta:")
    
    # Emoji para mas cute ang app
    emoji = "🐱" if predicted_class == "cats" else "🐶"
    
    st.success(f"{emoji} Predicted Class: **{predicted_class.upper()}**")
    st.info(f"Confidence Score: **{confidence:.2f}%**")
