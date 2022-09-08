import os
from mtcnn import MTCNN
import  streamlit as st
from PIL import Image
import pickle
import cv2
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from sklearn.metrics.pairwise import cosine_similarity



feature_list=np.array(pickle.load(open('features.pkl','rb')))
model=VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
detector = MTCNN()
filename=pickle.load(open('filename.pkl','rb'))
def save_uploaded_image(uploaded_img):
    try:
        with open(os.path.join('uploads',uploaded_img.name),'wb') as f:
            f.write(uploaded_img.getbuffer())
        return True
    except:
        return False

def extract_features(image_path,model,detector):

    # load Image and detect the face
    sample_img = cv2.imread(image_path)
    r = detector.detect_faces(sample_img)
    x, y, width, height = r[0]['box']
    face = sample_img[y:y + height, x:x + width]

    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image)
    face_array = face_array.astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocess_img = preprocess_input(expanded_img)
    result = model.predict(preprocess_img).flatten()
    return result



def recommend(feature_list,feature):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(feature.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    return sorted(list(enumerate(similarity)),reverse=True,key=lambda x:x[1])[0][0]

st.title("Which celebrity look like you")

uploaded_image=st.file_uploader("choose an image")
if uploaded_image is not None:
    # save image in directory
    if save_uploaded_image(uploaded_image):
        #load image
        st.text("uploaded Image")
        display_img=Image.open(uploaded_image)

        feature=extract_features(os.path.join('uploads',uploaded_image.name),model,detector)

        # st.text(feature)
        # st.text(feature.shape)
        index_pos=recommend(feature_list,feature)
        # st.text(index_pos)

        col1,col2=st.columns(2)
        with col1:
            st.header('Your Image')
            st.image(display_img)
        with col2:
            st.header(filename[index_pos].split('\\')[1])
            st.image(filename[index_pos],width=300)

