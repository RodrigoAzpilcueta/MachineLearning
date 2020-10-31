# IMPORTS NECESARIOS PARA QUE FUNCIONE
import os 
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import load_model 

def load_models():

    models = []
    models.append(load_model("models/model1.h5", compile=False))        
    models.append(load_model("models/model2.h5", compile=False))       
    models.append(load_model("models/model3.h5", compile=False))        
    models.append(load_model("models/model4.h5", compile=False))        
    models.append(load_model("models/model5.h5", compile=False))
    print("Load Models Ready")
    return(models)



def eval_video(file_path, models,r=None):

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    base_dir = os.path.dirname(file_path) + "/"
    out = np.array([])
    count = 0
    print("Reading Video...")
    frames = np.zeros((0,224,224,3))
    cap = cv2.VideoCapture(file_path)
    NUMBER_OF_FRAMES = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    while True:
        _, frame = cap.read()
        if r != None:
            frame = frame[r[1]:r[3], r[0]:r[2]]
            frame = cv2.resize(frame, (224,224))
            frame = np.expand_dims(frame, axis=0)
            frames = np.concatenate((frames, frame))
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == NUMBER_OF_FRAMES:
            cap.release()
            break
        frames = preprocess_input(frames)
        print("Frames extracted")
    for model in models:
        count += 1
        print("Evaluating model "+ str(count))
        result = model.predict(frames)
        result = np.expand_dims(result, axis=0)
        if out.size == 0:
            out = result
        else:
            out = np.concatenate((out, result))
    np.save(base_dir +  base_name + ".npy", out)
    print("Numpy Generated")
                

#models = load_models()
#path = 'D:\\Machine Learning\\Jupyter Files\\VideoTest\\Test/001.avi'
# eval_video( path,models, r= (162,33,527,402)) (X1,Y1,X2,Y2) X1,Y1 = Vertice superior izquierdo X2,Y2 = Vertice inferior derecho
