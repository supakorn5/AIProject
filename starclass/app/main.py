from fastapi import FastAPI, Request
import pickle
import requests

from app.code import Predict_star


app = FastAPI()
url = "http://172.17.0.3:80/api/getstar"
m = pickle.load(open(r'/projectAI/starclass/model/genstar.pkl', 'rb'))


@app.get("/")
def root():
    return {"message": "this is car api"}


@app.get("/api/starclass")
async def predict_chanel(request: Request):
    item = await request.json()
    item_img = item["img"]
    response = requests.get(url, json={"img": item_img})
    hog = response.json()
    # Extract the numerical values from the 'hog' dictionary and convert to an array
    hog_values = hog.get("Hog", [])  # Replace 'Hog' with the correct key
    hog_array = [float(val) for val in hog_values]

    # Ensure 'hog_array' is not empty before using it for prediction
    if not hog_array:
        print(hog_values)
        print(hog_array)
        return {"chanel": "Prediction data is empty."}

    chanel = Predict_star(m, hog_array)
    return {"chanel": chanel}
