sim_params = {
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

@dataclass
class Parameters:
    material = "brick"
    sampleLength = 0

class Simulation:
    """A simulation 
    
    Sialskdfas;


    """
    def __init__(self):
        self.sim_params = {
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

    def iterate(self):
        pass


def iterate(Sim: Simulation):
    """iterates
    
    Sialskdfas;

    Args:
        Sim ... a simulation object

    Returns:
        a numbers
    """
    pass

A = Simu