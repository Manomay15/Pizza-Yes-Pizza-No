import streamlit as st
from PIL import Image
from image_classification import teachable_machine_classification
st.title("Image Classification using neural networks")
st.header("Is it a pizza?")
st.text("Upload a food image:")

uploaded_file = st.file_uploader("Choose an image ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded picture.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(image, 'inceptionv3.h5')
    print(label)
    if label < 0.5:
        st.write("Yes, it's a pizza!!")
    else:
        st.write("Nope, my master won't like it :(")