import requests
from win10toast import ToastNotifier
import json
import time

def update() :
    r = requests.get("https://coronavirus-19-api.herokuapp.com/")
    data = r.json
    text = f'Case: {data["cases"]} \nDeaths:{data["deaths"]} \nRecovered : {data["recovered"]}'

    while True :
       t = ToastNotifier()
       t.show_toast("Covid 19 update", text, duration = 20)
       time.sleep(60)

update()