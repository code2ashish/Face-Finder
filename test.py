#load image -> face detection and extract its features
# find the cosine distance of current image with all the 8655 features
# recommend that image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import cv2
from mtcnn import MTCNN

feature_list=np.array(pickle.load(open('features.pkl','rb')))
filenames=pickle.load(open('filename.pkl','rb'))
model=VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

detector=MTCNN()
# load Image and detect the face
sample_img=cv2.imread('download.jpg')
r=detector.detect_faces(sample_img)
x,y,width,height=r[0]['box']
face=sample_img[y:y+height,x:x+width]


image=Image.fromarray(face)
image=image.resize((224,224))
face_array=np.asarray(image)
face_array=face_array.astype('float32')
expanded_img=np.expand_dims(face_array,axis=0)
preprocess_img=preprocess_input(expanded_img)
result=model.predict(preprocess_img).flatten()

# print(result)
# print(result.shape)
similarity=[]
for i in range(len(feature_list)):
    similarity.append(cosine_similarity(result.reshape(1,-1), feature_list[i].reshape(1,-1))[0][0])

final_output_idx=sorted(list(enumerate(similarity)),reverse=True,key=lambda x:x[1])[0][0]
img=cv2.imread(filenames[final_output_idx])
cv2.imshow('output',img)
cv2.waitKey(0)