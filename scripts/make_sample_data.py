"""Builds a tiny synthetic demo set so the pipeline runs end-to-end before real data arrives."""
import pandas as pd, os
rows = [
 ("No water supply in our street for three days", "en", "water", "high", "negative"),
 ("Big pothole on main road causing accidents", "en", "roads", "critical", "negative"),
 ("Street light not working and power cuts every night", "en", "power", "high", "negative"),
 ("Garbage not collected for a week, bad smell", "en", "sanitation", "high", "negative"),
 ("हमारे इलाके में कई दिनों से पानी नहीं आ रहा", "hi", "water", "high", "negative"),
 ("सड़क पर बड़ा गड्ढा है, दुर्घटना हो सकती है", "hi", "roads", "critical", "negative"),
 ("बिजली का तार टूटकर गिरा है, करंट का खतरा", "hi", "power", "critical", "negative"),
 ("कचरा उठाने वाली गाड़ी नहीं आती", "hi", "sanitation", "medium", "negative"),
 ("आमच्या भागात पाणी येत नाही", "mr", "water", "high", "negative"),
 ("रस्त्यावर खड्डे आहेत", "mr", "roads", "medium", "negative"),
]
df = pd.concat([pd.DataFrame(rows, columns=["text","language","department","urgency","sentiment"])]*6, ignore_index=True)
os.makedirs("data/samples", exist_ok=True)
df.to_csv("data/samples/grievances_sample.csv", index=False)
print("wrote", len(df), "rows")
