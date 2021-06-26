# MoistureTransport

Authors:

- Luo, Likun
- Holzner, Peter

A moisture transport simulation tool that simulates moisture absorption of a specimen/sample.

So far, the only material available is 'brick', though others (like cement) could likely easily be added

## Usage

### Source version

1. Clone the repository and use

2. Run using

```bash
./python main.py
```

### Release/Binary version

1. Download the binary

2. Run using

- Windows:

```bash
./moisture_transport.exe
```

- Linux:

```bash
./moisture_transport.exe
```

## Simulation parameters

The simulation can be fed with custom parameters of the sample - such as sample length or the mean pore size -, of the environment condition - such as the starting moisture saturation or free saturation - or for the numerical simulation - such as the number of elements used or the desired simulation time.

A set of all these parameters is referred to as a 'configuration' that is provided via a configuration file (in either YAML or JSON format) to the program.

See below or in the folder 'cfg' for examples.

### Example configuration

```YAML
material: brick # currently only brick is supported
sampleLength: 0.2 # in m
moistureUptakeCoefficient: 10.0 # in kg/(m^2 h^0.5)
freeSaturation: 300.0 # in kg/m^3
meanPoreSize: 1.e-2 # in m
freeParameter: 10
Anfangsfeuchte: 10 # in %
totalTime: 200 # in h
numberofElements: 28 # number of finite elements used
averagingMethod: "linear" # only linear supported for now
```

The configuration above can easily be expanded by developers looking to expand the current code - make sure to also add the parameter, its' type and bounds/choices to the `SettingsSchema`!
