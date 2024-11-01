import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import image

import deepface
from deepface import DeepFace
import gradio as gr
from fns.utility_fns import empty_img, make_records

def image_predict(mat_no_, student_name, img_):
    mat_no_ = mat_no_.upper()
    models = ['VGG-Face', 'Facenet', 'Facenet512', 'openFace', 'DeepFace', 'DeepId', 'ArcFace', 'Dlib', 'SFace']
    backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
    # df = pd.read_csv("records.csv")
    df = make_records()
    mat_nos = [i for i in df["matric number"].values]

    if mat_no_ in mat_nos:
        verified = True
    else:
        verified = False
    
    if verified:
        df_sort = df[df["matric number"] == mat_no_]
        imgs_ = df_sort["img paths"].values[0]
        imgs = imgs_.split(" ")
        # h_start = round(0.05*img_.shape[0])
        # h_end = round(0.95*img_.shape[0])
        # w_start = round(0.05*img_.shape[1])
        # w_end = round(0.95*img_.shape[1])
        # img_ = img_[h_start:h_end, w_start:w_end]
        verify_status = list()
        for img in imgs:
            result = DeepFace.verify(
                img1_path = img_,
                img2_path = img,
                model_name = models[1],
                distance_metric = 'cosine',
                enforce_detection = False,
                detector_backend = backends[-2],
                align = False,
                threshold = .2
            )
            verify_status.append(result["verified"])

        if True in verify_status:
            response_ = f"{student_name} is verified and can proceed to vote\n[Click the link to Vote:] ({'https://huggingface.co/spaces/AyoAgbaje/cast_vote'})"
            img_match_id = verify_status.index(True)
            img_match = imgs[img_match_id]
            img_match = image.imread(img_match)

            # return response_, img_match
        else:
            response_ = f"{student_name} cannot verified as image does not match image in the Database"
            img_match = empty_img()
            # return response_, img_match
    else:
        response_ = f"Matric number of the student:{student_name} is not found in Database"
        img_match = empty_img()
        
    return img_match, response_
    


with gr.Blocks() as demo:
    m_no = gr.Textbox(placeholder = "Input Matric Number in the format (DEPT/YY/NNNN) here:", label = "MATRIC NO")
    name_ = gr.Textbox(placeholder = "Input your name here", label = "Student Name".upper())
    image_ = gr.Image(label = 'Input Image to be verified', sources = "webcam")
    output1 = gr.Image(type = "filepath", label = "Database Image match")
    output2 = gr.Markdown(label = 'Verification Response')
    btn = gr.Button('Verify')
    btn.click(fn = image_predict, inputs = [m_no, name_, image_], outputs = [output1, output2])

demo.launch(share = True)