import cv2
import requests
import base64
# -*- coding: utf-8 -*-
import numpy as np
import json
img=cv2.imread('./val/Davis Alderete/310.png')
print(type(img))
code = base64.b64encode(img)
#print(type(code))
m={"msg":code.decode(),"shape":img.shape}
print(type(code.decode()))
s=json.dumps(m)
#print(s)
#res=requests.post("http://127.0.0.1:5000/showimage",json=s).json()
#print('this is res: ',res)