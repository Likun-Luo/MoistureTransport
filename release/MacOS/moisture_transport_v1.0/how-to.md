# How To

1. Edit the simulation configuration file found at `/cfg/input.yaml`.
    - Replacements with default values can be found in `cfg/defaults/`.
    - You can also copy the `input.yaml` file to save the default values.
    - If you make mistakes during editing, the program will (in 99% of cases) tell you (and also how to fix it).
    - Numerical values should be entered with `.`(dot) as floating point seperator
        - Values in scientific notation should be expressed as follows:
            - 1.e-2
            - 1.e+2
            - i.e. the `.`preceding the *e* and the sign following it should be included! (YAML limitations)
2. Start the program either by double clicking the executable
    - or start it from the terminal via `./mousture_transport`(or on Windows `./moisture_transport.exe`)
    - or start the script version via `python main.py`or `./main.py`
    - *ADVANCED*: If it is started from the terminal (either script or binary), additional advanced CL-arguments are available
        - query with `<script-or-binary> -h`
3. Follow the instructions given
    - It will ask you which configuration file to use (if you edited the default file, simply enter `y` and hit enter)
    - You can provide a path to a different file when prompted
