import random
import json
import pickle
import numpy as np
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("medical.json").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')


def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word)
                      for word in sentence_words]
    return sentence_words


def bagw(sentence):
    sentence_words = clean_up_sentences(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bagw(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res)
               if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]],
                            'probability': str(r[1])})
        return return_list


def context_storage(condition):
    f = open("variable_storage.txt", "w")
    f.write(condition)


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(message):

    # ints = predict_class(message)
    # res = get_response(ints, intents)
    # print(type(res))
    # print(res)
    # return res

    stored_message = message.split(" ")
    covid_words = ["covid", "Covid", "COVID-19", "Covid-19", "covid-19", "covid 19", "corona"]
    pots_words = ["pots", "POTS", "Pots", "Postural Orthostatic Tachycardia Syndrome"]
    if any(word in stored_message for word in covid_words):
        condition = "covid"
        context_storage("covid")
    if any(word in stored_message for word in pots_words):
        condition = "pots"
        context_storage("pots")

    variable_file = open("variable_storage.txt", 'r')
    condition = variable_file.readline()
    if len(message) > 15 and "how" not in message:
        message = message + " " + condition
    ints = predict_class(message.lower())
    res = get_response(ints, intents)

    if res[:2] == "00":

        if condition == "pots":
            if "managing_pots_symptoms" in res:
                return(
                    "There are many different ways to manage your symptoms:\n"
                    "exercise and physical activity can be helpful,\n"
                    "including walking and isometric exercises which contract your muscles while sitting or standing.\nAnother option of managing symptoms is to change positions very slowly, giving time to allow blood pressure to adjust.\nAn excellent way to alleviate some symptoms is adjusting diet and nutrition.\nTry increasing sodium intake from 3,000 mg to 10,000 mg a day, drinking 2 to 2.5 liters per day, and eating small frequent meals rather than 3 larger meals.\nAlso try to incorporate a diet with high fiber and complex carbohydrates.\nEating a balanced diet with protein, vegetables, dairy and fruit can also be beneficial.\nManaging your symptoms while sleeping requires a different set of tips.\nSome common suggestions are to try raising the head of your bed 6-10 inches to sleep at an angle, having a consistent room temperature, a consistent sleep schedule, avoiding daytime napping, and avoiding blue light before bed.\nTo prevent flare ups, you should try to maintain a consistent temperature, avoid prolonged standing, and avoid alcohol.")
            if "pots_symptom_list" in res:
                return(
                    "Symptoms of POTS include: Dizziness, Fainting Forgetfulness and Trouble Breathing,\nHeart Palpitations, Racing Heart, Exhaustion or Fatigue, Feeling Nervousness or Anxious, Shakiness or Excessive Sweating,\nShortness of Breath, Chest Pain, Headache, Feeling Sick, Bloating, Pale Face, Purple Discoloration of Hands and Feet, or Disrupted Sleep from Previously Listed Symptoms.")
            if "pots_information" in res:
                return(
                    "POTS stands for Postural Orthostatic Tachycardia Syndrome and it is a condition that causes a person’s\nheart to beat faster than normal when moving between standing, sitting or laying down. When a person without POTS does this,\ntheir body’s Autonomic Nervous System balances their heart rate and blood pressure to\nkeep a stable amount of blood flow to the brain. When a person with POTS does this, their body doesn’t\ncoordinate the constriction of their blood vessels meaning their brain does not have a steady blood flow.\nPOTS is a fairly common condition that affects approximately .2% of the population.\nBoth biological men and women can be affected but the majority of the people who\nare diagnosed are women between the ages of 15 and 50. Different stressors can increase \nyour chance of developing this such as, viral illnesses, serious infections, pregnancy, physical trauma, or surgery.\nOther things that increase your risk are certain autoimmune conditions such as lupus or celiac disease.")
            if "pots_resources" in res:
                return(
                    "Here are some resources that may help you with more information.\nCleveland Clinic:\nhttps://my.clevelandclinic.org/health/diseases/16560-postural-orthostatic-tachycardia-syndrome-pots\nNational Library of Medicine:\n https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7046364/#:\n~:text=The%20prevalence%20of%20POTS%20is,United%\n20States%20have%20the%20disorder\nMayo Clinic:\n https://www.mayoclinic.org/medical-professionals/endocrinology/news/postural-orthostatic-tachycardia-syndrome-and-chronic-fatigue-in-adolescents/mac-20430815\nNational Institute of Neurological Disorders and Stroke:\n https://www.ninds.nih.gov/health-information/disorders/postural-tachycardia-syndrome-pots\nJohn Hopkins Medicine:\n https://www.hopkinsmedicine.org/health/conditions-and-diseases/postural-orthostatic-tachycardia-syndrome-pots")
            if "pots_tests" in res:
                return(
                    "POTS is often diagnosed using the Tilt Table Test. During this test a person lays on a motorized table\nthat has straps to hold them in place. Equipment to measure blood pressure and heart rate are attached\non said person. The table will then begin to tilt into\ndifferent positions. If the person’s blood pressure decreases significantly or they faints then their medical provider may recommend\nother test such as the QSART, Autonomic Breathing Test, Tuberculin Skin Test, Skin Nerve Biopsy, Echocardiogram, Blood Volume,\nand/or blood and urine test to rule out conditions with similar symptoms.")
            if "pots_medication" in res:
                return(
                    "Medical professionals prescribe different kinds of medications for treatment. Some include beta blockers, midodrine, fludrocortisone, or a selective serotonin reuptake inhibitor. Talk to your provider to find the best fit for you.")
        elif (condition) == "covid":
            if "managing_covid_symptoms" in res:
                return(
                    "There are many different ways to manage your symptoms: \nThe effect individual should be to self isolate until it has been at least 5 days since their symptoms appeared, and you have had no fever for 24 hours\nwithout the help of fever-reducing medications.\nIf they are able, they should remain home. While at home, there are many different things they can do to improve their condition. To help with fever a person should drink lots of fluids and take acetaminophen (Tylenol). If suffering from a cough, they can try gargling salt water, adding honey to hot tea, or taking over the counter medicine if they have access to it. If at any point they are having trouble breathing, chest pain, or bluish lips and face immediately go to the hospital. While there, they will possibly use supplemental\noxygen, mechanical ventilation, or Extracorporeal membrane oxygenation")
            if "covid_information" in res:
                return(
                    "Coronavirus is a new virus to affect humans but has previously been found in bats, cats and camels. The virus would not infect the inhabited animals. As the virus was spread to different animals it mutated. The mutation that occurred when the virus was transferred to humans is what causes the infection.   The virus is spread through airborne droplets or the transfer of the virus from hands. Once the virus has entered a person it travels to the back of their nasal passages and the mucous membrane in the back of their throat. From there the virus multiplies and moves into the lungs and eventually spreads to the rest of the body.There are many different ways one can minimize their risk of developing this infection."
                )
            if "covid_symptom_list" in res:
                return(
                    "Symptoms of COVID-19 include: Fever or Chills, Cough, Shortness of Breath or Difficulty Breathing, Tiredness, Muscle or Body Aches, Headaches,\nNew Loss of Taste or Smell, Sore Throat, Congestion or Runny Nose, Nausea or Vomiting, or Diarrhea.")
            if "covid_resources" in res:
                return(
                    "Here are some resources that may help you with more information,\nCleveland Clinic: https://my.clevelandclinic.org/health/diseases/21214-coronavirus-covid-19\nSioux Center Health: https://www.siouxcenterhealth.org/covid-19-resources/how-do-i-manage-covid-19-symptoms-at-home/\nCDC: https://www.cdc.gov/coronavirus/2019-ncov/symptoms-testing/symptoms.html\nCOVID: https://www.covid.gov\nTexas Department of State Health Services: https://www.dshs.texas.gov/covid-19-coronavirus-disease-2019")
            if "avoiding_covid" in res:
                return(
                    "To avoid COVID-19 a person should wash their hands regularly, wear a face mask, avoid touching their eyes, mouth,\nor nose, cover their mouth and nose when sneezing, avoid large crowds and close contact with others, clean surfaces frequently, and\nstrengthen their immune system. Strengthening a person’s immune system is done by having a consistent sleep\nschedule, a healthy diet, drinking lots of fluids, and exercising.")
            if "covid_tests" in res:
                return(
                    "COVID-19 is diagnosed through a laboratory or at home test using a saliva sample, a nose or throat swab.\nA person should be tested if they are suddenly feeling sick or have come in close contact with a person known or\nsuspected to have tested positive. If you believe you may have COVID-19 but receive a\nnegative test result you have a false negative. This could be due to testing too early or your sample being defective\ndue to not being swabbed deep enough or the mishandling of the sample.")
            if "covid_medication" in res:
                return(
                    "Medical professionals will prescribe antivirals to help people who are infected. Some of these antivirals are Nirmatrelvir\nwith Ritonavir (Paxlovid), Remdesivir (Veklury), or Molnupiravir (Lagevrio).\nTalk to your provider to find the best fit for you.")
        else:
            return "Sorry I'm not really sure what condition you're talking about. Mind trying again?"

    else:
        return res


# print(chatbot_response("Hello"))

from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == "__main__":
    app.run()