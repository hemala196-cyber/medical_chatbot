from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pyttsx3   # ← ADD THIS

medicine_info = {
    "dolo 650": "Dolo 650 is used to reduce fever and relieve mild to moderate pain.",
    "amlodipine": "Amlodipine is used to treat high blood pressure and chest pain.",
    "pan d": "Pan D is used for acidity, heartburn, and stomach issues.",
    "rantac": "Rantac helps reduce acid production in the stomach.",
    "cetrizine": "Cetrizine is used to relieve allergy symptoms like sneezing and itching."
}

# 1. TRAIN NLP MODEL
commands = list(medicine_info.keys())
labels = [1, 1, 1, 0, 0]

cv = CountVectorizer()
x = cv.fit_transform(commands)

model = LogisticRegression()
model.fit(x, labels)

# 2. TEXT INPUT
def listen():
    text = input("Enter medicine name: ")
    return text.lower()

# 3. PREDICT MEDICINE
def get_medicine_description(text):
    text_vector = cv.transform([text])
    prediction = model.predict(text_vector)

    if prediction[0] == 1:
        for med in medicine_info:
            if med in text:
                return medicine_info[med]

    return "Sorry, medicine not recognized."

# 4. TEXT TO SPEECH
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# 5. MAIN PROGRAM
spoken_text = listen()

if spoken_text:
    description = get_medicine_description(spoken_text)
    print(description)
    speak(description)
else:
    speak("I could not understand. Please try again.")
