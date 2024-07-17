#ChatBotAI Speech Recognition Vasileios Vasiloudis

#Python Libraries
import numpy
import time
import pyttsx3
import tflearn
import tensorflow
import random
import json
import wave
import pyaudio
import geocoder
from geopy.geocoders import Nominatim
from pydub import AudioSegment
from pydub.playback import play
import nltk
nltk.download("punkt")
from nltk.stem.lancaster import LancasterStemmer
import speech_recognition as sr

# Translate the location to english language
greek_alphabet = "ΑαΆάΒβΓγΔδΕεΈέΖζΗηΉήΘθΙιΊίΚκΛλΜμΝνΞξΟοΌόΠπΡρΣσςΤτΥυΎύΦφΧχΨψΩωΏώ"
latin_alphabet = "AaAaBbGgDdEeEeZzHhHhJjIiIiKkLlMmNnXxOoOoPpRrSssTtUuUuFfQqYyWwWw"
greek2latin = str.maketrans(greek_alphabet, latin_alphabet)

# Parameters for Recording
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
seconds = 5


# Parameters about engine to speak the machine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
# female voice
engine.setProperty('voice', 'english+f1')
engine.setProperty('voice', 'english+f2')
engine.setProperty('voice', 'english+f3')
engine.setProperty('voice', 'english+f4')
engine.setProperty('voice', 'english_rp+f3')
engine.setProperty('voice', 'english_rp+f4')
volume = engine.getProperty('volume')
engine.setProperty('volume', 10.0)
rate = engine.getProperty('rate')
engine.setProperty('rate', rate - 25)

# opening JSON file, the dataset
file = open("First_Aid_dataset.json")

# returns JSON object as a directionary
data = json.load(file)

# Variables
words = []
labels = []
docs_x = []
docs_y = []
stemmer = LancasterStemmer()

# iterating through the json list
for intent in data['intents']:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

# Starting Training
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)

# Build ChatBotAI
class ChatBot():
   while True:
       r = sr.Recognizer()
       with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.2)
            print("Hello I am First Aid ChatBot Christine, how can i help you?")
            question1 = "Hello I am First Aid ChatBot Christine, how can i help you?"
            engine.say(question1)
            engine.runAndWait()
            time.sleep(2)
            audio2 = r.listen(source)
            try:
                print("Processeing...")
                question2 = "Processeing..."
                engine.say(question2)
                engine.runAndWait()
                time.sleep(2)
                inp = r.recognize_google(audio2)
                inp = inp.lower()
                results = model.predict([bag_of_words(inp,words)])[0]
                results_index = numpy.argmax(results)
                tag = labels[results_index]

                if results[results_index] > 0.5:
                    for tg in data["intents"]:
                        if tg['tag'] == tag:
                            responses = tg['responses']
                            print(random.choice(responses))
                            text = random.choice(responses)
                            engine.say(text)
                            engine.runAndWait()
                            print("Now tell me in exact description your wound or pain so I can take it right to the rescuer" )
                            answer1 = "Now tell me in exact description your wound or pain so I can take it right to the rescuer"
                            engine.say(answer1)
                            engine.runAndWait()
                            # Start Recording
                            print("Start Recording...")
                            start_record = "Start Recording"
                            engine.say(start_record)
                            engine.runAndWait()
                            frames = []
                            p = pyaudio.PyAudio()

                            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
                            for i in range(0, int(RATE / CHUNK * seconds)):
                                data = stream.read(CHUNK)
                                frames.append(data)
                            time.sleep(2)
                            print("Recording stopped")
                            stop_record = "Recording stopped"
                            engine.say(stop_record)
                            engine.runAndWait()
                            stream.stop_stream()
                            stream.close()
                            p.terminate()
                            wf = wave.open("output.wav", "wb")
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(p.get_sample_size(FORMAT))
                            wf.setframerate(RATE)
                            wf.writeframes(b''.join(frames))
                            wf.close()
                            # The ChatBot wait 5 second to play the recording
                            time.sleep(5)
                            record = AudioSegment.from_wav("output.wav")
                            play(record)
                            # Gps
                            print("The specific coordinates and exact address from injured are as follows:")
                            answer2 = "The specific coordinates and exact address from injured are as follows:"
                            engine.say(answer2)
                            engine.runAndWait()
                            g = geocoder.ip("me")
                            print(g.latlng)
                            geo1 = g.latlng
                            engine.say(geo1)
                            engine.runAndWait()
                            location = Nominatim(user_agent="GatLoc")
                            locationName = location.reverse(g.latlng)
                            print("Address:", locationName.address.translate(greek2latin))
                            geo2 = locationName.address.translate(greek2latin)
                            engine.say(geo2)
                            engine.runAndWait()
                            # Exit ChatBot
                            exit()
            except sr.UnknownValueError:
                print("I did not get that, try again")
                answer3 = "I did not get that, try again"
                engine.say(answer3)
                engine.runAndWait()


# Function who start the ChatBot
ChatBot()
