# ADELAIDE
_Autonomous Digital Everyday Liaison and Artificially Intelligent Digital Entity_ 

This project is being developed in sections with a focus on developing my skills in different areas of tech, ultimately serving as an intro in AI/ML.

### Instructions for use
- To run eyes or brain:
     - **Create a Python 3.9.6 conda env**
     - pip install -r requirements.txt
     - brew install portaudio
     - python eyes.py/brain.py
          - *brain.py requires an active AssemblyAi API key and paid access to their Streaming-Speech-To-Text API

### Roadmap

- **ADELAIDE's Eyes** _Initial build complete_
     - Implementing OpenCV object detection using YOLO for use in future features.
     - **Technologies**: _OpenCV_, _Ultralytics_
     - **Languages**: _Python_

- **ADELAIDE's Brain** _Initial build complete_
     - Implementing an LLM framework with a TTS framework for vocal interaction with ADELAIDE
     - **Technologies**: _LLAMA 3_, _AssemblyAI Stream-Text-To-Speech_
     - **Notes**: ***Constants*** **file with API keys necessary for build. If you wish to clone this repo you must obatain your own API keys**
     - Designing and implementing databases for memories, learning, and other future use cases within the project

- **Integrating ADELAIDE with her Environment**
     - Following vision and LLM implementation I plan to train ADELAIDE on the environment, i.e. facial data, routines, security protocols, etc.Ultimately this step integrates the LLM, CV, NN, and any other database frameworks used.

- **Home Integration**
     - The goal of this project is chiefly, to learn and to have a cool home assistant. But, given the scope of ADELAIDE, and the capabilities of what is planned, I believe this product could ultimately be of use to more than just myself, and I would love to see how far I can go with it!
