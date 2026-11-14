"""Rule-based generator for labelled complaints in English, Hindi and Marathi.

Every complaint is assembled from independent, grammatically complete sentences
(issue + optional location + optional duration + optional closing), so no
agreement errors cross sentence boundaries. Labels come from the issue's severity,
nudged up by long duration or an urgent closing line. Rows are tagged
label_source="generated" and must be reported as such.
"""
from __future__ import annotations

import random

import pandas as pd

LEVELS = ["low", "medium", "high", "critical"]
DEPTS = ["water", "roads", "power", "sanitation", "other"]
DEPT_WEIGHTS = [0.22, 0.22, 0.20, 0.22, 0.14]
LEVEL_WEIGHTS = [0.2, 0.4, 0.3, 0.1]

# ---- issues: lang -> dept -> {level index: [sentences]} -------------------------------
EN = {
 "water": {0: ["The water pressure from our tap is lower than usual.", "The public water tap near our lane drips continuously."],
           1: ["The water supply in our area is irregular.", "The tap water is cloudy and has a bad smell.", "Water tanker timings are not being followed."],
           2: ["There has been no water supply to our house.", "A main water pipeline has burst and water is being wasted on the road.", "Our colony gets no water in the morning hours at all."],
           3: ["Sewage is mixing with the drinking water and people are falling sick.", "A burst pipeline has flooded the street and entered several homes."]},
 "roads": {0: ["The footpath tiles along our street are broken.", "The road markings near the school have faded."],
           1: ["There are many potholes on the main road.", "The road surface has cracked after the rains.", "A speed breaker on our street is damaged."],
           2: ["A large pothole on the road is causing vehicles to skid.", "Road repair work was left half-finished and traffic is blocked."],
           3: ["An open manhole on the road could cause a fatal accident.", "The bridge railing is broken and there is a risk of someone falling."]},
 "power": {0: ["One street light on our lane is not working.", "The street light timer is faulty and the lamp stays on during the day."],
           1: ["We face power cuts every day in our locality.", "The voltage is very low in the evenings.", "The electricity meter reading in our building seems wrong."],
           2: ["The transformer has failed and the whole area has no electricity.", "Our area has had a power outage since last night."],
           3: ["A live electric wire has fallen on the road and may cause electrocution.", "The transformer is sparking and has caught fire."]},
 "sanitation": {0: ["The public dustbin near our lane is full.", "Street sweeping is skipped on our road sometimes."],
           1: ["The garbage collection vehicle does not come regularly.", "Our street has not been swept for several days.", "Garbage is being dumped on the empty plot next to us."],
           2: ["A pile of garbage has been lying here and is causing a foul smell.", "The drain is blocked and dirty water is flowing onto the road."],
           3: ["The sewer is overflowing and dirty water is entering homes.", "Medical waste has been dumped openly near the school."]},
 "other": {0: ["I cannot find the status of my birth certificate application.", "The property tax receipt on the portal shows the wrong name."],
           1: ["The trees in the nearby park have not been trimmed.", "The number of stray dogs in our area has increased.", "Loudspeaker noise continues late at night in our neighbourhood."],
           2: ["Illegal construction has blocked the access road to our lane.", "A public park gate is broken and unsafe for children."],
           3: ["A part of an old building near the bus stop is at risk of collapsing."]},
}
HI = {
 "water": {0: ["नल से आने वाले पानी का दबाव थोड़ा कम है।", "हमारी गली का सार्वजनिक नल लगातार टपक रहा है।"],
           1: ["हमारे इलाके में पानी की सप्लाई अनियमित है।", "नल से गंदा पानी आ रहा है।", "पानी के टैंकर समय पर नहीं आ रहे हैं।"],
           2: ["हमारे घर में पानी नहीं आ रहा है।", "मुख्य पाइपलाइन फट गई है और सड़क पर पानी बह रहा है।"],
           3: ["पीने के पानी में सीवर का पानी मिल रहा है और लोग बीमार पड़ रहे हैं।", "पाइपलाइन फटने से गली में पानी भर गया है और कई घरों में घुस गया है।"]},
 "roads": {0: ["हमारी गली का फुटपाथ टूटा हुआ है।", "स्कूल के पास सड़क की सफेद पट्टियाँ मिट गई हैं।"],
           1: ["मुख्य सड़क पर कई गड्ढे हैं।", "बारिश के बाद सड़क में दरारें पड़ गई हैं।", "हमारी गली का स्पीड ब्रेकर टूट गया है।"],
           2: ["सड़क पर बड़ा गड्ढा है जिससे वाहन फिसल रहे हैं।", "सड़क की मरम्मत अधूरी छोड़ दी गई है और यातायात रुका हुआ है।"],
           3: ["सड़क पर खुला मैनहोल है और किसी की जान जा सकती है।", "पुल की रेलिंग टूट गई है और गिरने का खतरा है।"]},
 "power": {0: ["हमारी गली की एक स्ट्रीट लाइट बंद है।", "स्ट्रीट लाइट का टाइमर खराब है और बत्ती दिन में भी जलती रहती है।"],
           1: ["हमारे मोहल्ले में रोज बिजली कटौती हो रही है।", "शाम के समय बिजली का वोल्टेज बहुत कम रहता है।", "हमारी बिल्डिंग के बिजली मीटर की रीडिंग गलत लग रही है।"],
           2: ["ट्रांसफार्मर खराब हो गया है और पूरे इलाके की बिजली गुल है।", "कल रात से हमारे इलाके में बिजली नहीं है।"],
           3: ["बिजली का तार टूटकर सड़क पर गिरा हुआ है और करंट का खतरा है।", "ट्रांसफार्मर में चिंगारियाँ निकल रही हैं और आग लग गई है।"]},
 "sanitation": {0: ["हमारी गली का सार्वजनिक कूड़ेदान भर गया है।", "हमारी सड़क पर कभी-कभी झाड़ू नहीं लगती।"],
           1: ["कचरा उठाने वाली गाड़ी नियमित रूप से नहीं आती।", "कई दिनों से हमारी गली में सफाई नहीं हुई है।", "बगल के खाली प्लॉट में कचरा फेंका जा रहा है।"],
           2: ["यहाँ कचरे का ढेर पड़ा है और बदबू फैल रही है।", "नाली जाम है और गंदा पानी सड़क पर बह रहा है।"],
           3: ["सीवर ओवरफ्लो हो रहा है और गंदा पानी घरों में घुस रहा है।", "स्कूल के पास खुले में मेडिकल कचरा फेंका गया है।"]},
 "other": {0: ["मेरे जन्म प्रमाण पत्र के आवेदन की स्थिति नहीं मिल रही है।", "पोर्टल पर प्रॉपर्टी टैक्स की रसीद में नाम गलत दिख रहा है।"],
           1: ["पास के पार्क के पेड़ों की छंटाई नहीं हुई है।", "हमारे इलाके में आवारा कुत्तों की संख्या बढ़ गई है।", "हमारे मोहल्ले में देर रात तक लाउडस्पीकर का शोर रहता है।"],
           2: ["अवैध निर्माण से हमारी गली का रास्ता बंद हो गया है।", "सार्वजनिक पार्क का गेट टूटा है और बच्चों के लिए असुरक्षित है।"],
           3: ["बस स्टॉप के पास एक पुरानी इमारत का हिस्सा गिरने का खतरा है।"]},
}
MR = {
 "water": {0: ["नळाला पाण्याचा दाब थोडा कमी आहे.", "आमच्या गल्लीतील सार्वजनिक नळ सतत गळत आहे."],
           1: ["आमच्या भागात पाणीपुरवठा अनियमित आहे.", "नळाला गढूळ पाणी येत आहे.", "पाण्याचे टँकर वेळेवर येत नाहीत."],
           2: ["आमच्या घरी पाणी येत नाही.", "मुख्य जलवाहिनी फुटली आहे आणि रस्त्यावर पाणी वाहत आहे."],
           3: ["पिण्याच्या पाण्यात सांडपाणी मिसळत आहे आणि लोक आजारी पडत आहेत.", "जलवाहिनी फुटल्याने गल्लीत पाणी साचले आहे आणि अनेक घरांत शिरले आहे."]},
 "roads": {0: ["आमच्या गल्लीतील पदपथ तुटलेला आहे.", "शाळेजवळील रस्त्यावरील पांढरे पट्टे पुसले गेले आहेत."],
           1: ["मुख्य रस्त्यावर अनेक खड्डे आहेत.", "पावसानंतर रस्त्याला भेगा पडल्या आहेत.", "आमच्या गल्लीतील गतिरोधक तुटला आहे."],
           2: ["रस्त्यावर मोठा खड्डा आहे आणि वाहने घसरत आहेत.", "रस्त्याचे काम अर्धवट सोडले आहे आणि वाहतूक अडली आहे."],
           3: ["रस्त्यावर उघडे मॅनहोल आहे आणि कोणाचाही जीव जाऊ शकतो.", "पुलाचे कठडे तुटले आहेत आणि पडण्याचा धोका आहे."]},
 "power": {0: ["आमच्या गल्लीतील एक पथदिवा बंद आहे.", "पथदिव्याचा टायमर बिघडला आहे आणि दिवा दिवसाही चालू राहतो."],
           1: ["आमच्या वस्तीत रोज वीज जाते.", "संध्याकाळी विजेचा दाब खूप कमी असतो.", "आमच्या इमारतीच्या वीज मीटरचे रीडिंग चुकीचे वाटते."],
           2: ["रोहित्र बिघडले आहे आणि संपूर्ण परिसरातील वीज गेली आहे.", "काल रात्रीपासून आमच्या भागात वीज नाही."],
           3: ["विजेची तार तुटून रस्त्यावर पडली आहे आणि शॉक लागण्याचा धोका आहे.", "रोहित्रातून ठिणग्या उडत आहेत आणि आग लागली आहे."]},
 "sanitation": {0: ["आमच्या गल्लीतील सार्वजनिक कचराकुंडी भरली आहे.", "आमच्या रस्त्यावर कधीकधी झाडलोट होत नाही."],
           1: ["कचरा गाडी नियमित येत नाही.", "अनेक दिवसांपासून आमच्या गल्लीत स्वच्छता झालेली नाही.", "शेजारच्या रिकाम्या जागेत कचरा टाकला जात आहे."],
           2: ["येथे कचऱ्याचा ढीग पडला आहे आणि दुर्गंधी पसरत आहे.", "गटार तुंबले आहे आणि घाण पाणी रस्त्यावर वाहत आहे."],
           3: ["मलनिःसारण वाहिनी ओसंडून वाहत आहे आणि घाण पाणी घरांत शिरत आहे.", "शाळेजवळ उघड्यावर वैद्यकीय कचरा टाकला आहे."]},
 "other": {0: ["माझ्या जन्म दाखल्याच्या अर्जाची स्थिती समजत नाही.", "पोर्टलवरील मालमत्ता कराच्या पावतीवर नाव चुकीचे दिसत आहे."],
           1: ["जवळच्या उद्यानातील झाडांची छाटणी झालेली नाही.", "आमच्या भागात भटक्या कुत्र्यांची संख्या वाढली आहे.", "आमच्या वस्तीत रात्री उशिरापर्यंत ध्वनिक्षेपकाचा आवाज चालू असतो."],
           2: ["बेकायदेशीर बांधकामामुळे आमच्या गल्लीचा रस्ता बंद झाला आहे.", "सार्वजनिक उद्यानाचे प्रवेशद्वार तुटले आहे आणि मुलांसाठी असुरक्षित आहे."],
           3: ["बस थांब्याजवळील एका जुन्या इमारतीचा भाग कोसळण्याचा धोका आहे."]},
}
# ---- neutral requests / positive feedback: lang -> kind -> dept -> [sentences] -------------
EN_NP = {"neutral": {"water": ["Please share the water supply timings for our area.", "I would like to know how to apply for a new water connection."],
                     "roads": ["Please tell me when the road resurfacing in our area is scheduled.", "Whom should I contact for a road repair request?"],
                     "power": ["Please confirm how to apply for a street light at our colony entrance.", "I would like to know the process to shift my electricity meter."],
                     "sanitation": ["Please share the garbage collection schedule for our ward.", "How can I request a new dustbin for our lane?"],
                     "other": ["Please tell me how to get a copy of my birth certificate.", "I would like to know the procedure to report stray animals."]},
         "positive": {"water": ["Thank you for restoring the water supply so quickly.", "The pipeline was repaired promptly and we are grateful."],
                      "roads": ["Thank you for filling the potholes on our road.", "The road repair was completed well and on time."],
                      "power": ["Thank you for fixing the street light in our lane.", "The electricity was restored quickly, thanks to the team."],
                      "sanitation": ["Thank you for clearing the garbage from our street.", "The sweeping staff has been very regular lately, thank you."],
                      "other": ["Thank you for trimming the trees in our park.", "My certificate was issued on time, thank you."]}}
HI_NP = {"neutral": {"water": ["कृपया हमारे क्षेत्र में पानी की सप्लाई का समय बताएं।", "मुझे नया पानी कनेक्शन लेने की प्रक्रिया जाननी है।"],
                     "roads": ["कृपया बताएं कि हमारे इलाके की सड़क की मरम्मत कब होगी।", "सड़क मरम्मत के अनुरोध के लिए किससे संपर्क करें?"],
                     "power": ["कृपया बताएं कि कॉलोनी के गेट पर स्ट्रीट लाइट लगवाने के लिए आवेदन कैसे करें।", "मुझे बिजली मीटर शिफ्ट कराने की प्रक्रिया जाननी है।"],
                     "sanitation": ["कृपया हमारे वार्ड में कचरा उठाने का समय बताएं।", "हमारी गली के लिए नया कूड़ेदान कैसे मांगें?"],
                     "other": ["मुझे जन्म प्रमाण पत्र की कॉपी प्राप्त करने की प्रक्रिया जाननी है।", "आवारा पशुओं की शिकायत दर्ज करने की प्रक्रिया क्या है?"]},
         "positive": {"water": ["पानी की सप्लाई इतनी जल्दी बहाल करने के लिए धन्यवाद।", "पाइपलाइन की मरम्मत जल्दी हो गई, हम आभारी हैं।"],
                      "roads": ["हमारी सड़क के गड्ढे भरने के लिए धन्यवाद।", "सड़क की मरम्मत समय पर और अच्छी तरह हुई।"],
                      "power": ["हमारी गली की स्ट्रीट लाइट ठीक करने के लिए धन्यवाद।", "बिजली जल्दी बहाल हो गई, टीम का धन्यवाद।"],
                      "sanitation": ["हमारी गली से कचरा साफ करने के लिए धन्यवाद।", "सफाई कर्मचारी आजकल नियमित आ रहे हैं, धन्यवाद।"],
                      "other": ["पार्क के पेड़ों की छंटाई करने के लिए धन्यवाद।", "मेरा प्रमाण पत्र समय पर जारी हो गया, धन्यवाद।"]}}
MR_NP = {"neutral": {"water": ["कृपया आमच्या भागातील पाणीपुरवठ्याच्या वेळा कळवाव्यात.", "नवीन नळजोडणीसाठी अर्ज कसा करायचा ते मला जाणून घ्यायचे आहे."],
                     "roads": ["कृपया आमच्या भागातील रस्त्याची दुरुस्ती कधी होणार ते सांगा.", "रस्ते दुरुस्तीच्या विनंतीसाठी कोणाशी संपर्क साधावा?"],
                     "power": ["वसाहतीच्या प्रवेशद्वारावर पथदिवा बसवण्यासाठी अर्ज कसा करावा ते कळवा.", "वीज मीटर हलवण्याची प्रक्रिया मला जाणून घ्यायची आहे."],
                     "sanitation": ["कृपया आमच्या प्रभागातील कचरा संकलनाच्या वेळा कळवाव्यात.", "आमच्या गल्लीसाठी नवीन कचराकुंडी कशी मागावी?"],
                     "other": ["जन्म दाखल्याची प्रत मिळवण्याची प्रक्रिया मला जाणून घ्यायची आहे.", "भटक्या प्राण्यांची तक्रार नोंदवण्याची प्रक्रिया काय आहे?"]},
         "positive": {"water": ["पाणीपुरवठा इतक्या लवकर सुरू केल्याबद्दल धन्यवाद.", "जलवाहिनीची दुरुस्ती लगेच झाली, आम्ही आभारी आहोत."],
                      "roads": ["आमच्या रस्त्यावरील खड्डे बुजवल्याबद्दल धन्यवाद.", "रस्त्याची दुरुस्ती वेळेत आणि चांगली झाली."],
                      "power": ["आमच्या गल्लीतील पथदिवा दुरुस्त केल्याबद्दल धन्यवाद.", "वीज लवकर सुरू झाली, टीमचे आभार."],
                      "sanitation": ["आमच्या गल्लीतील कचरा साफ केल्याबद्दल धन्यवाद.", "सफाई कर्मचारी अलीकडे नियमित येत आहेत, धन्यवाद."],
                      "other": ["उद्यानातील झाडांची छाटणी केल्याबद्दल धन्यवाद.", "माझे प्रमाणपत्र वेळेवर मिळाले, धन्यवाद."]}}

PLACES = {"en": ["Sector 12", "Gandhi Nagar", "MG Road area", "Ward 7", "Shivaji Nagar", "Rajiv Nagar", "Lake View Colony", "Block C", "Ashok Vihar", "Ganesh Nagar"],
          "hi": ["सेक्टर 12", "गांधी नगर", "एमजी रोड क्षेत्र", "वार्ड 7", "शिवाजी नगर", "राजीव नगर", "लेक व्यू कॉलोनी", "ब्लॉक सी", "अशोक विहार", "गणेश नगर"],
          "mr": ["सेक्टर 12", "गांधी नगर", "एमजी रोड", "वॉर्ड 7", "शिवाजी नगर", "राजीव नगर", "लेक व्ह्यू कॉलनी", "ब्लॉक सी", "अशोक विहार", "गणेश नगर"]}
PROB_LOC = {"en": "This problem is in {p}.", "hi": "यह समस्या {p} में है।", "mr": "ही समस्या {p} परिसरात आहे."}
LIVE_LOC = {"en": "We live in {p}.", "hi": "हम {p} में रहते हैं।", "mr": "आम्ही {p} परिसरात राहतो."}
DURATION = {"en": "This has been going on for {n} days.", "hi": "यह समस्या {n} दिनों से बनी हुई है।", "mr": "ही समस्या {n} दिवसांपासून सुरू आहे."}
CLOSE_PLAIN = {"en": ["Please take action.", "Kindly resolve this at the earliest."],
               "hi": ["कृपया कार्रवाई करें।", "कृपया इसे जल्द से जल्द हल करें।"],
               "mr": ["कृपया कार्यवाही करावी.", "कृपया हे लवकरात लवकर सोडवावे."]}
CLOSE_URGENT = {"en": "This is urgent as children and elderly people live here.",
                "hi": "यह बहुत जरूरी है क्योंकि यहाँ बच्चे और बुजुर्ग रहते हैं।",
                "mr": "हे अत्यावश्यक आहे कारण येथे लहान मुले आणि वृद्ध राहतात."}
CLOSE_ANGRY = {"en": "This is completely unacceptable.", "hi": "यह बिल्कुल अस्वीकार्य है।", "mr": "हे अगदी अस्वीकारार्ह आहे."}

ISSUES = {"en": EN, "hi": HI, "mr": MR}
NP = {"en": EN_NP, "hi": HI_NP, "mr": MR_NP}


def _complaint(lang: str, rng: random.Random) -> dict:
    dept = rng.choices(DEPTS, DEPT_WEIGHTS)[0]
    base = rng.choices(range(4), LEVEL_WEIGHTS)[0]
    level = base
    parts = [rng.choice(ISSUES[lang][dept][base])]
    if rng.random() < 0.6:
        parts.append(PROB_LOC[lang].format(p=rng.choice(PLACES[lang])))
    if rng.random() < 0.45 and base < 3:
        n = rng.randint(2, 10)
        parts.append(DURATION[lang].format(n=n))
        if n >= 5 and level <= 1:
            level += 1
    roll = rng.random()
    if roll < 0.2 and base < 3:
        parts.append(CLOSE_URGENT[lang]); level = max(level, min(level + 1, 2))
    elif roll < 0.35:
        parts.append(CLOSE_ANGRY[lang])
    elif roll < 0.75:
        parts.append(rng.choice(CLOSE_PLAIN[lang]))
    return {"text": " ".join(parts), "language": lang, "department": dept,
            "urgency": LEVELS[level], "sentiment": "negative"}


def _non_complaint(lang: str, rng: random.Random) -> dict:
    kind = rng.choice(["neutral", "positive"])
    dept = rng.choices(DEPTS, DEPT_WEIGHTS)[0]
    parts = [rng.choice(NP[lang][kind][dept])]
    if rng.random() < 0.5:
        parts.append(LIVE_LOC[lang].format(p=rng.choice(PLACES[lang])))
    return {"text": " ".join(parts), "language": lang, "department": dept, "urgency": "low", "sentiment": kind}


def generate(n_en: int = 420, n_hi: int = 300, n_mr: int = 280, seed: int = 42, non_complaint_share: float = 0.15) -> pd.DataFrame:
    """Unique, shuffled rows with label_source='generated'."""
    rng = random.Random(seed)
    rows, seen = [], set()
    for lang, n in (("en", n_en), ("hi", n_hi), ("mr", n_mr)):
        made, tries = 0, 0
        while made < n and tries < n * 50:
            tries += 1
            r = _non_complaint(lang, rng) if rng.random() < non_complaint_share else _complaint(lang, rng)
            if r["text"] in seen:
                continue
            seen.add(r["text"]); rows.append(r); made += 1
    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    df["label_source"] = "generated"
    return df
