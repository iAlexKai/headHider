from aip import AipSpeech
import requests
import json
import speech_recognition as sr
import re
# import win32com.client

# 初始化语音
# 2、音频文件转文字：采用百度的语音识别python-SDK
# 导入我们需要的模块名，然后将音频文件发送给出去，返回文字。
# 百度语音识别API配置参数
# speaker = win32com.client.Dispatch("SAPI.SpVoice")
APP_ID = '20572331'
API_KEY = 'W8BBz3yQhWj3aHMvjXPg8GU0'
SECRET_KEY = 'VYsbIEbuaGEeZuzidIHn6bOSHG5tFSnY'
client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
path = 'voices/myvoices.wav'

# headers:
# target_path = '/Users/alex/Library/Caches/Transmit/57F10EE5-C6B7-451B-AA72-83342306E1E3/10.105.242.52/home/liu/myk/cvaed_torch/content_from_local.txt'
# cvaed
target_path = '/Users/alex/Library/Caches/Transmit/DDD03A4B-B061-438A-AEDB-2617B9F398D6/10.105.242.52/home/liu/myk/before_20201026/poetry/cvaed/content_from_local.txt'

# 3、与机器人对话：调用的是图灵机器人
# 图灵机器人的API_KEY、API_URL
turing_api_key = "c7c4f9bf6e8f443eaada5de8bfdb1bc7"
api_url = "http://openapi.tuling123.com/openapi/api/v2"  # 图灵机器人api网址
headers = {'Content-Type': 'application/json;charset=UTF-8'}

# 1、语音生成音频文件,录音并以当前时间戳保存到voices文件中
# Use SpeechRecognition to record 使用语音识别录制
def my_record(rate=16000):
    r = sr.Recognizer()
    with sr.Microphone(sample_rate=rate) as source:
        print("听你说~")
        audio = r.listen(source)

    with open(path, "wb") as f:
        f.write(audio.get_wav_data())


# 将语音转文本STT
def listen():
    # 读取录音文件
    with open(path, 'rb') as fp:
        voices = fp.read()
    try:
        # 参数dev_pid：1536普通话(支持简单的英文识别)、1537普通话(纯中文识别)、1737英语、1637粤语、1837四川话、1936普通话远场
        result = client.asr(voices, 'wav', 16000, {'dev_pid': 1537, })
        # result = CLIENT.asr(get_file_content(path), 'wav', 16000, {'lan': 'zh', })
        # print(result)
        # print(result['result'][0])
        # print(result)
        result_text = result["result"][0]
        if "藏头诗" in result_text:
            with open(target_path, 'w') as file:
                words = re.sub(r'[^\u4e00-\u9fa5]','',result_text[result_text.index("诗")+1:])
                file.write(words[:6])
                print("藏头诗： " + words[:4])
        elif "古体诗" in result_text:
            with open(target_path, 'w') as file:
                words = re.sub(r'[^\u4e00-\u9fa5]','',result_text[result_text.index("诗")+1:])
                file.write(words[:6])
                print("诗题： " + words[:4])
        else:
            # print(result_text)
            pass

        return result_text[0:-1]
    except KeyError:
        print("没听到声音")
        return " "


# 图灵机器人回复
def Turing(text_words=""):
    req = {
        "reqType": 0,
        "perception": {
            "inputText": {
                "text": text_words
            },

            "selfInfo": {
                "location": {
                    "city": "北京",
                    "province": "北京",
                    "street": "车公庄"
                }
            }
        },
        "userInfo": {
            "apiKey": turing_api_key,  # 你的图灵机器人apiKey
            "userId": "Nieson"  # 用户唯一标识(随便填, 非密钥)
        }
    }

    req["perception"]["inputText"]["text"] = text_words
    response = requests.request("post", api_url, json=req, headers=headers)
    response_dict = json.loads(response.text)

    result = response_dict["results"][0]["values"]["text"]
    print("AI Robot said: " + result)
    return result


class Robot:
    def __init__(self):
        self.mode = 0
        self.myCommands = {
            "开机": self.start,
            "关机":self.stop,
            "前进":self.go_forward(),
            "后退":self.back_forward(),
        }

    def start(self):
        self.mode = 1
        return "欸，我在"

    def stop(self):
        self.mode = 0
        return "拜拜主人"

    def go_forward(self):
        return "执行前进命令"

    def back_forward(self):
        return "执行后退命令"


if __name__ == '__main__':
    # 语音合成，输出机器人的回答
    my_robot = Robot()
    while True:
        my_record()
        request = listen()

        for key in my_robot.myCommands.keys():
            if key in request:
                word = my_robot.myCommands[key]()
                # speaker.Speak("执行用户命令" + word)
                break

        if my_robot.mode == 1:
            response = Turing(request)
            # speaker.Speak(response)





