from tensorflow.keras.models import load_model
import cv2
import numpy as np
img = cv2.imread("/Users/mac/Desktop/TESR/Contents/teachable machine/test2.jpeg") #path image

sizeTarget = (224, 224)

np.set_printoptions(suppress=True)
dataObj = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

model = load_model("/Users/mac/Desktop/TESR/Contents/teachable machine/keras_model.h5") #path model

if img is not None:
    img_resize = cv2.resize(img,sizeTarget) #resize image
    
    image_array = np.asarray(img_resize)#convert image to array

    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1 #normalized image

    dataObj[0] = normalized_image_array #get frist dimention 
    prediction =  list(model.predict(dataObj)[0])#change np.ndarray to list 
    idx = prediction.index(max(prediction)) #get index is maximun value

    if idx == 0:
        cv2.putText(img, "pfizer: "+str(round(prediction[idx]*100,2)) +"%", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2,cv2.LINE_AA)
    elif idx == 1:
        cv2.putText(img, "astra: "+str(round(prediction[idx]*100,2))+"%", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2,cv2.LINE_AA)
    elif idx == 2:
        cv2.putText(img, "moderna: "+str(round(prediction[idx]*100,2))+"%", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2,cv2.LINE_AA)
    elif idx == 3:
        cv2.putText(img, "sinovac: "+str(round(prediction[idx]*100,2))+"%", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2,cv2.LINE_AA)


    cv2.imshow("Predict Result", img) #image show
    k = cv2.waitKey(0)#wait all key for close window 


cv2.destroyAllWindows()