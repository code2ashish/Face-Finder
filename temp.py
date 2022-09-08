import os
import cv2
import  streamlit as st
img=st.file_uploader("Upload")

def save_uploaded_image(uploaded_img):

        with open(os.path.join('uploads',uploaded_img.name),'wb') as f:
            f.write(uploaded_img.getbuffer())
        return True


# img=cv2.imread('img.jpg')
if img is not None:
    if save_uploaded_image(img):
        st.image(img)
        st.text("succ")