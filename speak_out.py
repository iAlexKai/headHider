import pyttsx3 
import os

engine = pyttsx3.init() 

last_text = None
while True:
    try:
        if os.path.exists('./content_from_remote.txt'):
            os.system('rm ./content_from_remote.txt')
        
        # headers
        # os.system('wget http://10.105.242.52:8000/myk/cvaed_torch/content_from_remote.txt')
        # cvaed
        os.system('wget http://10.105.242.52:8000/myk/before_20201026/poetry/cvaed/content_from_remote.txt')
        
        with open('./content_from_remote.txt', 'r') as file:
            content = file.read()
            # import pdb
            # pdb.set_trace()
            if last_text == content:
                continue
            last_text = content
            engine.say(content)

            engine.runAndWait()
    except:
        continue
