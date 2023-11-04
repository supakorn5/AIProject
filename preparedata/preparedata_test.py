import cv2
import requests
import base64
import os
import pickle

url = "http://localhost:8080/api/getstar"


def img2vec(img):
    v, buffer = cv2.imencode(".jpg", img)
    img_str = base64.b64encode(buffer)
    data = "image data," + str.split(str(img_str), "'")[1]
    response = requests.get(url, json={"img": data})
    return response.json()


path = r'E:\projectAI\preparedata\dataset\test'
list_data = []
i = 0
for sub in os.listdir(path):
    for fn in os.listdir(os.path.join(path, sub)):
        img_file_name = os.path.join(path, sub)+"/"+fn
        img = cv2.imread(img_file_name)
        res = img2vec(img)
        vec = list(res["Hog"])
        vec.append(i)
        list_data.append(vec)
        print(i, os.path.join(path, sub)+"/"+fn)
    i = i+1


write_path = r"E:\projectAI\preparedata\model\star_test.pkl"
pickle.dump(list_data, open(write_path, "wb"))
print("data preparation test is done")
