#create newapp
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import time
import glob
import random
import re
import os
import tempfile
import ssl
import math

st.title('Action Recognition Model For Babies')
st.text('This model was developed by Victor Adewopo, Nelly Elsayed. \nSoIT-University of Cincinnati in collaboration with Procter and Gamble.')


# Or even better, call Streamlit functions inside a "with" block:

st.write('Only five  action can be recognized by our trained model')
actions=('on_feet','active','rest', 'escape','crawling')
st.write(actions)

st.write('')
#st.text('Video should be maximum of 15 seconds')
#st.write('.')
vidup= st.file_uploader('Video should be maximum of 15 seconds. \nUpload your video below')

PATH=os.getcwd()
#st.write('Loading saved model') 

#Load Saved Model
#@st.cache
def K_I_512D(videopath):
    K_I_512D=tf.keras.models.load_model(PATH +'/saved_model/5sec_AR_kineticsweightsonly_noflatten_moredense.h5')
    v=K_I_512D.predict(videopath)
    return v
#@st.cache
def K_I_aug_20E(videopath):
    K_I_aug_20E=tf.keras.models.load_model(PATH +'/saved_model/augumented_5sec_AR_kinetics+ImageNetweightsonly.h5')
    v=K_I_aug_20E.predict(videopath)
    return v
#@st.cache
def K_3DL(videopath):
    K_3DL=tf.keras.models.load_model(PATH +'/saved_model/5sec_AR_kineticsweightsonly_noflatten_moredense.h5')
    v=K_3DL.predict(videopath)
    return v

#st.write('All models loaded') 
    

#For streamlt
#Instant prediction without trimnming or saving the junks
def all_embed(videopath):
    
    def dummy_get_totalframes_fps(videopath):      #Get duration of clips function #Changing this script to read fs and total 
        #frames at once because of stream lit bug that wants me to realodd the page twice before getting them
        clip= cv2.VideoCapture(videopath)
        totalframes = int(clip.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(clip.get(cv2.CAP_PROP_FPS))
        totalframes=int(totalframes)
        #st.write(fps,totalframes)
        return[totalframes, fps, clip]
    def get_totalframes(path):
        tframes=Frame_and_fps[0]
        return tframes
    def get_5secs(path):
        fps=Frame_and_fps[1]
        return fps
    #@st.cache(allow_output_mutation=True)
    def get_clip(path):
        clip=Frame_and_fps[2]
        return clip
    
    
    def get_range(framestart,frameend,vidfps): #This function to get prediction range
        prediction_range=('{} - {}'.format(math.ceil(framestart/vidfps),math.ceil(frameend/vidfps)))
        #st.write('\n')
        st.write('Action Detected for %s seconds is'%(prediction_range))
        #st.write('------------------------------------------------------------------') 

    def video_test(videopath,framestart,frameend): ##Instant prediction without trimnming or saving the junks

        def crop_center_square(frame):
            (y, x) = frame.shape[0:2]
            min_dim = min(y, x)
            start_x = x // 2 - min_dim // 2
            start_y = y // 2 - min_dim // 2
            return frame[start_y:start_y + min_dim, start_x:start_x
                         + min_dim]

        cap = cv2.VideoCapture(videopath)
        max_frames = 450#int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # limiting the frame to 450 which is equivalent of 15 seconds
        frames = []
        frames = np.zeros(shape=(max_frames, 224, 224, 3))
        i=0
        try:
            while True:
                (ret, frame) = cap.read()
                if not ret:
                    break
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, (224, 224))
                frame = frame[:, :, [2, 1, 0]]

                frames[i]=frame
                i+=1
                if i==max_frames:
                    break

        finally:
            cap.release()
        newframes1=frames[framestart:frameend]
        newframes=np.pad(newframes1, ((0,150-len(newframes1)),(0,0),(0,0),(0,0)), mode='constant')
        return (tf.constant(newframes, dtype=tf.float32)[tf.newaxis, ...])/ 255.0
    
    def modelpredict(clip): #Get the model output for all 5sec junk
       # print(clip.shape)
        labels=np.array(['on_feet', 'active', 'rest', 'escape', 'crawling'])
        K_I_512D_result=K_I_512D((clip))[0]
        probabilities1 = tf.nn.softmax(K_I_512D_result).numpy()
        probabilities1=np.array(probabilities1)
        #st.write('------------------------------------------------------------------') 
        
        for i in np.argsort(probabilities1)[::-1][:3]:
            st.write("\t\t-->{}".format(f"  {labels[i]:}: {probabilities1[i] * 100:5.2f}%"))
            #st.write(f"  {labels[i]:}: {probabilities1[i] * 100:5.2f}%")
        #st.write('------------------------------------------------------------------') 

            #st.write(f"  {labels[i]:}: {probabilities3[i] * 100:5.2f}%")
        st.write('------------------------------------------------------------------')
    
    def split_frames(videopath): # breaks all long video into short frames and prints the results
        tframes=get_totalframes(videopath)
       # st.write(tframes)
        fivesecs=get_5secs(videopath)*5
        vidfps=get_5secs(videopath)
        #st.write(vidfps)
        divisorrate=round(tframes/fivesecs) #Get the divsion rate to know how many number of iterations will be done on the video
        framestart=0
        frameend=fivesecs
        st.write('The uploaded video will be splitted to :', divisorrate,'chunks of 5-seconds each.')
        st.write('------------------------------------------------------------------') 
        for i in range(divisorrate):
            if tframes-frameend>fivesecs:
                get_range(framestart,frameend,vidfps)
                modelpredict(video_test(videopath,framestart,frameend))
            elif tframes-frameend<0:
                framestart=(tframes-fivesecs)
                frameend=(framestart+fivesecs)
                get_range(framestart,frameend,vidfps)
                modelpredict(video_test(videopath,framestart,frameend))
            else:
                framestart=(tframes-fivesecs)-(tframes-frameend)
                get_range(framestart,frameend,vidfps)
                modelpredict(video_test(videopath,framestart,frameend))
            framestart=frameend
            frameend=framestart+fivesecs
    if (dummy_get_totalframes_fps(videopath)[0])< 450:
        Frame_and_fps=dummy_get_totalframes_fps(videopath)       
        split_frames(videopath)
        st.video(videopath)
    else: # I am limiting this app to 15 seconds because of memory will crash on streamlit if larger.
        st.write('Uploaded file is greater than 15 seconds. Consider uploading shorter video')

if vidup is not None:
    videofeed=vidup.read()
    vidhold=tempfile.NamedTemporaryFile(delete=False)
    vidhold.write(videofeed)
    #upload_details={'video_type':vidup.type,'video_name':vidup.name}
    #st.write(upload_details)

    videopath=vidhold.name
    all_embed(videopath)
#    for files in os.listdir(PATH+'/tempfile/'):
 #       os.remove(PATH+'/tempfile/'+files)
   
