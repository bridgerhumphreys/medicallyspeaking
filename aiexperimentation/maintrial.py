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

covid = {"general_info": "", "symptoms": "", "symptom_management": "", "condition_management": "", "more_resources": ""}
pots = {"general_info": "", "symptoms": "", "symptom_management": "", "condition_management": "", "more_resources": ""}

# defining context variable for storage
condition = ""


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


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


print(
    "Hello, I'm Medically Speaking, a chatbot AI designed to assist you with any medical questions or concerns you might have. How can I help?")

condition = None

while True:
    message = input("")
    stored_message = message.split(" ")
    covid_words = ["covid", "Covid", "COVID-19", "Covid-19", "covid-19", "covid 19", "corona"]
    pots_words = ["pots", "POTS", "Pots", "Postural Orthostatic Tachycardia Syndrome"]
    if any(word in stored_message for word in covid_words):
        condition = "covid"

    if any(word in stored_message for word in pots_words):
        condition = "pots"

    ints = predict_class(message.lower())
    res = get_response(ints, intents)

    if res[:2] == "00":

        if condition == "pots":
            if "managing_pots_symptoms" in res:
                print(
                    "There are many different ways to manage your symptoms:\n1. Exercise and Physical Activity can be helpful.\nThis includes walking and Isometic Exercise which is contracting your muscles while sitting or standing. Another physical way to to slowing change posisions to allow blood pressure to adjust.\n2. Another excellent way to help with symptoms is adjusting diet and nutrion\nTry increasing sodium intake from 3,000mg to 10,000mg a day, drinking 2 to 2.5 liters per day, eat small frequent meals rather than 3 larger meals.\nAlso try to incorporate a diet with high fiber and complex carbohydrates.\nEat a balance diet with protein, vegetables, dairy and fruit.\n3. Try monitoring your pulse and blood prsssure.\nDo this by this information down and sharing it with your medical provider.\nManaging your symptoms while sleeping requires a different set of tips\nTry raising the head of your bed 6-10 inches to sleep at an angle, having a consistent room temperature, a consistent sleep schedule, avoid day time napping, and avoid blue light before bed.\nTo prevent flare ups you should try to maintain a consistent temperature, avoid prolonged standing and avoid alcohol.")
            if "pots_symptom_list" in res:
                print(
                    "Symptoms of POTS include: Dizziness, Fainting Forgetfulness and Trouble Breathing,\nHeart Palpitations, Racing Heart, Exhaustion/Fatigue, Feeling Nervousness/Anxious, shakiness and Excessive Sweating,\nShortness of Breath, Chest Pain, Headache, Feeling Sick, Bloating, Pale Face, Purple Discoloration of Hands and Feet,n or Disrupted Sleep from previously listed symptoms.")
            if "pots_information" in res:
                print(
                    "POTS stands for Postural Orthostatic Tachycardia Syndrome and it is a condition that causes a person’s\nheart to beat faster than normal when standing, sitting or laying down. When a person without POTS does this,\ntheir body’s Autonomic Nervous System to balance their heart rate and blood pressure to\nkeep a stable amount of blood flow to the brain. When a person with POTS does this, their body doesn’t\ncoordinate the constriction of their blood vessels meaning their brain does not have a steady blood flow.\nPOTS is a fairly common condition that affects approximately .2% of the population.\nBoth biological men and women can be affected but the majority of the people who\nare diagnosed are females between the ages of 15 and 50. Different stressors can increase \nyour chance of developing this such as, viral illnesses  or serious infections, pregnancy, physical trauma, or surgery.\nOther things that increase your risk are certain autoimmune conditions such as lupus or celiac disease.")
            if "pots_resources" in res:
                print(
                    "Here are some resources that may help you with more information,\nCleveland Clinic:https://my.clevelandclinic.org/health/diseases/16560-postural-orthostatic-tachycardia-syndrome-pots\nNational Library of Medicine: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7046364/#:~:text=The%20prevalence%20of%20POTS%20is,United%20States%20have%20the%20disorder\nMayo Clinic: https://www.mayoclinic.org/medical-professionals/endocrinology/news/postural-orthostatic-tachycardia-syndrome-and-chronic-fatigue-in-adolescents/mac-20430815\nNational Institute of Neurological Disorders and Stroke: https://www.ninds.nih.gov/health-information/disorders/postural-tachycardia-syndrome-pots\nJohn Hopkins Medicine: https://www.hopkinsmedicine.org/health/conditions-and-diseases/postural-orthostatic-tachycardia-syndrome-pots")
            if "pots_tests" in res:
                print(
                    "POTS is often diagnosed using the Tilt Table Test. During this test a person lays on a motorized table\nthat has straps to hold them in place. Both electrodes that measure a person’s heart rate are attached\nand a blood pressure monitor are placed on said person. The table will then begin to tilt into\ndifferent positions. If the person’s blood pressure decreases or faints then the medical provider may recommend\nother test such as the QSART, Autonomic Breathing Test, Tuberculin Skin Test, Skin Nerve Biopsy, Echocardiogram, Blood Volume,\nand/or blood and urine test to rule out conditions with similar symptoms.")
            if "pots_medication" in res:
                print("")
        if condition == "covid":
            if "managing_covid_symptoms" in res:
                print(
                    "There are many different ways to manage your symptoms: Your first step should be to self isolate until it has been at least 5\ndays since your symptoms appeared and they are improving AND you have had no fever for 24 hours\nwithout the help of fever-reducing medications. If you can, you should try to stay home. As you are at home there are many\ndifferent things you can do to improve your condition. To help with fever you should drink\nlots of fluids and take acetaminophen (Tylenol). If you are suffering from a cough you can try gargling\nsalt water, adding honey to hot tea or hot water, avoid laying on your back, and taking over the counter\nmedicine if you have access to it. If at any point you are having trouble breathing, chest pain,\nor bluish lips and face immediately go to the hospital. While there they will possible use supplemental\noxygen, mechanical ventilation, or Extracorporeal membrane oxygenation")
            if "covid_symptom_list" in res:
                print(
                    "Symptoms of COVID-19 include: Fever or Chills, Cough, Shortness of Breath or Difficulty Breathing, Tiredness, Muscle or Body Aches, Headaches,\nNew Loss of Taste or Smell, Sore Throat, Congestion or Runny Nose, Nausea or Vomiting, or Diarrhea.")
            if "covid_information" in res:
                print(
                    "Coronavirus is a new virus to affect humans but has previously been found in bats, cats and camels. The virus would not infect the\ninhabited animals. As the virus was spread to different animals it mutated. The mutation that occurred when the virus\nwas transferred to humans is what causes the infection.   The virus is spread through airborne\ndroplets or the transfer of the virus from hands. Once the virus has entered a person it travels\nto the back of their nasal passages and the mucous membrane in the back of their throat. From there the virus multiplies\nand moves into the lungs and eventually spreads to the rest of the body.There are many different ways\none can minimize their risk of developing this infection.")
            if "covid_resources" in res:
                print(
                    "Here are some resources that may help you with more information,\nCleveland Clinic: https://my.clevelandclinic.org/health/diseases/21214-coronavirus-covid-19\nSioux Center Health: https://www.siouxcenterhealth.org/covid-19-resources/how-do-i-manage-covid-19-symptoms-at-home/\nCDC: https://www.cdc.gov/coronavirus/2019-ncov/symptoms-testing/symptoms.html\nCOVID: https://www.covid.gov\nTexas Department of State Health Services: https://www.dshs.texas.gov/covid-19-coronavirus-disease-2019")
            if "avoiding_covid" in res:
                print(
                    "To avoid COVID-19 a person should wash their hands regularly, wear a face mask, avoid touching your eyes, mouth,\nor nose, cover your mouth and nose when sneezing, avoid large crowds and close contact, clean surfaces frequently, and\nstrengthen their immune system. Strengthening a person’s immune system is done by having a consistent sleep\nschedule, a healthy diet, drinking lots of fluids, and exercising.")
            if "covid_tests" in res:
                print(
                    "COVID-19 is diagnosed through a laboratory or at home test using a saliva sample, a nose or throat swab.\nA person should be tested if they are suddenly feeling sick or have come in close contact with a person known or\nsuspected to have tested positive. If you believe you may have COVID-19 but receive a\nnegative test result you have a false negative. This could be due to testing too early or your sample being defective\ndue to not being swabbed deep enough or the mishandling of the sample.")
            if "covid_medication" in res:
                print(
                    "Medical professionals will prescribe antivirals to help people who are infected. Some of these antivirals are Nirmatrelvir\nwith Ritonavir (Paxlovid), Remdesivir (Veklury), or Molnupiravir (Lagevrio).\nTalk to your provider to find the best fit for you.")

    else:
        print(res)




