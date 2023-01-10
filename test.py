# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests

article = "Die schwedische Klimaaktivistin Greta Thunberg verzichtet auf die Teilnahme am diesjährigen Weltklimagipfel, der am 6. November in Ägypten beginnt. Das kündigte die 19-Jährige nach einem Bericht der britischen Zeitung »The Guardian« am Sonntagabend bei der Vorstellung eines Buches zu den Folgen der Erderwärmung in London an. Thunberg kritisierte die als COP27 bezeichnete Klimakonferenz als Forum zum »Greenwashing«. Mit dem Begriff ist das Vortäuschen von erfolgreichen Maßnahmen zum Umweltschutz gemeint mit dem Ziel, umwelt- und verantwortungsbewusst zu erscheinen. Die Klimagipfel seien »nicht wirklich dazu gedacht, das ganze System zu ändern«, sondern ermutigten stattdessen zu allmählichem Fortschritt, sagte sie. »So wie sie sind, funktionieren die COPs nicht wirklich, es sei denn, wir nutzen sie als Gelegenheit zur Mobilisierung.« Als Grund für ihr Fernbleiben nannte sie auch eingeschränkte Möglichkeiten für zivilgesellschaftliche Beteiligung in Ägypten. »Ich gehe aus vielen Gründen nicht zur COP27, aber der Raum für die Zivilgesellschaft ist in diesem Jahr extrem beschränkt.« Darüber hinaus verzichtet Thunberg zum Wohle des Klimas auf Flugreisen. Ägypten ist aus Schweden ohne Flugzeug nur schwierig zu erreichen. Die 27. Weltklimakonferenz findet vom 6. bis 18. November im ägyptischen Scharm el-Scheich statt. Dort wird es vor allem um eine größere Unterstützung der wohlhabenden und für einen Großteil der klimaschädlichen CO2-Emissionen verantwortlichen Länder für ärmere Staaten gehen."


model_inputs = {
    "prompt": f"[Text]: {article} \n\n[Titel]: ",
    "temperature": 1.0, 
    "top_p": 0.75,
    "num_beams": 1,
    "do_sample": True
  }

res = requests.post('http://localhost:8000/', json = model_inputs)

print(res.json())