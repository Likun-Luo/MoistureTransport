import json

obj = {
	"material": "brick",
	"sampleLength": 0.2,
	"moistureUptakeCoefficient": 10.0,
	"freeSaturation": 300.0,
	"meanPoreSize": 10e-6,
	"freeParameter": 10,
	"numberofElements": 100,
	"timeStepSize": 0.01,
	"totalTime": 10,
	"Anfangsfeuchte": 40
}

sim_obj = {
	"L": 0.2,
}

class Parser:

    def get(self):
        return json.load(path)