
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import os
import argparse



parse=argparse.ArgumentParser()
parse.add_argument('-s','--source',default='1',help="Ingrese en que recurso desea usar para la inferencia, 0 para camara web y 1 para una imagen")
parse.add_argument('-i','--image',help="ingrese la ruta de la imagen en la que desea ejecutar el algoritmo")
args=parse.parse_args()
source=args.source
image_path=args.image


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,keep_all=False,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

resnet = InceptionResnetV1(pretrained='vggface2').eval()

workers = 0 if os.name == 'nt' else 4

load_data=torch.load('data.pt')
embedding_list=load_data[0]
name_list=load_data[1]



if source=='0': #face recognition running in real time
    print('opening web camera...')

    def recognition(img,emb_list):
        
        img_cropped_list,prob=mtcnn(img,return_prob=True)
        
        boxes,_=mtcnn.detect(img)
        if img_cropped_list is not None:

            if prob>0.80:
                    img_embedding=resnet(img_cropped_list.unsqueeze(0)).detach()
                    dist_list=[]
                        
                    for idx,emb_db in enumerate(emb_list):
                        dist=torch.dist(img_embedding,emb_db).item()
                        dist_list.append(dist)
                    min_dist=min(dist_list)
                    min_dist_idx=dist_list.index(min_dist)
                    name=name_list[min_dist_idx]

                    box=boxes[0]
                    original_img=img.copy()
                    if min_dist<0.90:
                        img=cv2.putText(img,name+' '+str(min_dist),(int(box[0]),int(box[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
                    img=cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),2)
            
        return img

    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        if not ret:
            print("fail to grab frame, try again")
            break
        img=recognition(frame,embedding_list)
        cv2.imshow('webCam',img)
        if cv2.waitKey(20)==27:
            break

    capture.release()
    cv2.destroyAllWindows()

else: #face recognition running in a simple image, source == 1
    
    print("running in a simple image")
    img=cv2.imread(image_path)
    img_cropped,prob=mtcnn(img,return_prob=True)


    boxes,_=mtcnn.detect(img)

    if prob>0.90:
        img_embedding=resnet(img_cropped.unsqueeze(0)).to(device)
        dist_list=[]
        for idx,emb_db in enumerate(embedding_list):
            dist=torch.dist(img_embedding,emb_db).item()
            dist_list.append(dist)
        min_dist=min(dist_list)
        min_dist_idx=dist_list.index(min_dist)
        name=name_list[min_dist_idx]

        box=boxes[0]
        original_img=img.copy()
        if min_dist<0.90:
            img=cv2.putText(img,name+' '+str(min_dist),(int(box[0]),int(box[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
        img=cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),2)



    while True:
        cv2.imshow("Smarcity ",img)
        if cv2.waitKey(20)==27:
            break
    
    cv2.destroyAllWindows() # destroy all windows


