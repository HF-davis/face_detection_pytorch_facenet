from flask import Flask,send_file,jsonify
from flask import request
import json
import os
import base64
import numpy as np
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

PEOPLE_FOLDER = os.path.join('static', 'people_photo')
app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER


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

@app.route('/')
def hello_world():
    return '<div>Hola mundo</div>'

@app.route('/data',methods=['GET'])
def Data():
    return {
        'name':'Davis',
        'password':'1234'
    }

@app.route('/Inference',methods=['GET'])
def inference():
    path='./static/people_photo/user.png'
    if os.path.exists(path):
        img=cv2.imread(path)
        img_cropped,prob=mtcnn(img,return_prob=True)
        boxes,_=mtcnn.detect(img)
        if img_cropped is not None:
        
            if prob>0.90:
                img_embedding=resnet(img_cropped.unsqueeze(0)).to(device)
                dist_list=[]
                for idx,emb_db in enumerate(embedding_list):
                    dist=torch.dist(img_embedding,emb_db).item()
                    dist_list.append(dist)
                min_dist=min(dist_list)
                min_dist_idx=dist_list.index(min_dist)
                name=name_list[min_dist_idx]

                #box=boxes[0]
                #original_img=img.copy()
                #if min_dist<0.90:
                #    img=cv2.putText(img,name+' '+str(min_dist),(int(box[0]),int(box[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
                #img=cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),2)
                print('im here 1')
                #data=base64.b64encode(img)
                return jsonify({'name':name, 'id':1})
        print('im here 2')
        #data=base64.b64encode(img)
        return jsonify({'name':'Nobody', 'id':2})
    return json.dumps("false")
        

@app.route('/SavePhoto',methods=['POST'])
def RealTime():
    
    img_data=request.json['img']
    with open("./static/people_photo/user.png","wb") as f:
        f.write(base64.b64decode(img_data.split(',')[1]))
    
    return json.dumps("Image saved succesfully")

if __name__=='__main__':
    app.run(debug=True)