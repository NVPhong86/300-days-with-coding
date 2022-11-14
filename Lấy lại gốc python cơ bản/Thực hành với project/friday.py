import pyttsx3
import datetime
import speech_recognition as sr
import webbrowser as wb
import os

friday = pyttsx3.init()
voice = friday.getProperty('voices')
friday.setProperty('voice',voice[1].id) # voice[0].id : giong nam ; voice[1].id , giong nu

def speak(audio) :
    print("F.R.I.D.A.D.A.Y : " + audio)
    friday.say(audio)
    friday.runAndWait()

def time() :
    Time = datetime.datetime.now().strftime("%I: %M :%p")
    speak(time)
def welcome() :
    hour = datetime.datetime.now().hour
    if hour >=6 and hour <12 :
        speak("Good morning")
    elif  hour >=12 and hour <18 :
        speak('Good Afternoon')
    else :
        speak ('Good night my baby')

speak("How can I help you ")

def command() :
    c = sr.Recognizer()
    with sr.Microphone as source :
        c.pause_threshold = 2 
        audio = c.listen(source)
    try :
        query = c.recognize_google(audio,language='en')
        print("Van Phong" + query)
    except sr.UnknownValueError :
        print ("Please repeat or typing the commnad ")
        query = str(input("Your typing is : "))
    return query

if __name__ == "__main__" :
    welcome()
    while True :
        query = command().lower()
        if "google" in query :
            speak("what should i search Phong : ")
            search = command().lower()
            url = f"https://www.google.com/search?q={search}"
            wb.get().open(url)
            speak(f'Here is your {search} on google')

        if "youtube" in query :
            speak("what should i search Phong : ")
            search = command().lower()
            url = f"https://www.youtube.com/search?q={search}"
            wb.get().open(url)
            speak(f'Here is your {search} on youtube')

        elif "open video " in query :
            link_video = []
            os.startfile(link_video)
        elif "quit" in query :
            speak("Good bye quynh")
            quit()





# pip install speech_recognition
# pip install pyaudio
