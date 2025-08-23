import requests

sample = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22.0,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
}

try:
    resp = requests.post("http://127.0.0.1:8000/predict", json=sample, timeout=10)
    print("Status code:", resp.status_code)
    print("Response:", resp.json())
except Exception as e:
    print("Error calling API:", e)
