import base64
import torch
import open_clip
from PIL import Image
from mobileclip.modules.common.mobileone import reparameterize_model
import google.genai as genai
from openai import OpenAI
from groq import Groq
from openrouter import OpenRouter
import json
import threading
import os

online=True
multiple=True
responses = []
model_path = "C:\\Users\\mihir\\source\\repos\\Image Recognition\\mobileclip2_s0.pt"

#add your api keys here
google_api_key = "AIzaSyAVvc7DJkK3gk40yQ6EPnUDb-jEoY4WhNU"
grok_api_key = "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"
router_api_key = "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"

#model names for online mode
#multiple models are used in order to create a layer on top of all of them and combine their results, if preferred, set multiple=False to use only one for quicker run time  
google_model_name = "gemini-2.5-flash"
grok_model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
router_model_name = "qwen/qwen2.5-vl-72b-instruct"


prompt = """

    You are a trash classification AI.

    Classify the object into one of these categories:
    - wet waste such as food waste like banana peels leftovers and organic scraps
    - plastic waste such as plastic bottles containers wrappers and packaging trash
    - metal waste such as metal cans aluminum foil and metallic trash

    Provide your answer in the following JSON format:
    Category should be one of "wet/plastic/metal" and confidence should be a number between 0 and 100 indicating the confidence level of the classification.

    {
        "category": "wet/plastic/metal",
        "confidence": 92
    }
"""


    #mobileclip

#loading the local model, you can change the model name and path as per your requirements, for example you can use "MobileCLIP2-S3" or "MobileCLIP2-S4" or any of the L-14 models, if using those models make sure to remove the model_kwargs as they are not needed for those models
model_name = "MobileCLIP2-S0" #name of your mobileclip model
model_path = "C:\\Users\\mihir\\source\\repos\\Image Recognition\\mobileclip2_s0.pt" # path location of your mobileclip model
model_kwargs = {}
if not (model_name == "MobileCLIP2-S3" or model_name == "MobileCLIP2-S4" or model_name.endswith("L-14")):
    model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=model_path, **model_kwargs)
tokenizer = open_clip.get_tokenizer(model_name)
model.eval()
model = reparameterize_model(model)



def local():
    values=[] #wet,plastic,metal
    image = preprocess(Image.open("C:\\Users\\mihir\\source\\repos\\Image Recognition\\trash.png").convert("RGB")).unsqueeze(0)
    text = tokenizer([
        "food waste like banana peels leftovers and organic scraps",
        "plastic bottles containers wrappers and packaging trash",
        "metal cans aluminum foil and metallic trash"
    ])
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1) 
        for i,value in enumerate(text_probs[0]):
            values.append(round(value.item()*100,2))
        return values

def google(prompt,key,Model):
    try:
        client = genai.Client(api_key=key)
        image = Image.open("trash.png")
        google_response = client.models.generate_content(model=Model,contents=[prompt,image])
        responses.append(json.loads(google_response.text[8:-3]))
        print("google confirmed")
    except:
        return

def openai(prompt,key,Model):
    try:
        client = OpenAI(api_key=key)
        image = Image.open("trash.png")
        with open("trash.png", "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        openai_response = client.responses.create(
            model=Model,
            input=[
                {
                    "role": "user",
                    "content": [
                        { "type": "input_text", "text":prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{image_base64}",
                        },
                    ],
                }
            ],
        )
        responses.append(json.loads(openai_response[8:-3]))
        print("openai confirmed")
    except:
        return

def grok(prompt, key, Model):
    try:
        client = Groq(api_key=key)
        with open("trash.png", "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        chat_completion = client.chat.completions.create(
            model=Model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )

        print("grok confirmed")
        responses.append(json.loads(chat_completion.choices[0].message.content[4:-4]))
    except:
        return

def router(prompt, key, Model):
    try:
        client = OpenRouter(api_key=key)
        with open("trash.png", "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        response = client.chat.send(
            model=Model,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )
        print("router confirmed")
        responses.append(json.loads(response.choices[0].message.content[8:-3]))
    except:
        return

if online ==False:
    print(local())
else:
    values = [0,0,0] #wet/plastic/metal
    if multiple:
        router_response = threading.Thread(target=router, args=(prompt, router_api_key, router_model_name))
        google_response = threading.Thread(target=google, args=(prompt, google_api_key, google_model_name))
        grok_response = threading.Thread(target=grok, args=(prompt, grok_api_key, grok_model_name))

        router_response.start()
        google_response.start()
        grok_response.start()

        router_response.join()
        google_response.join()
        grok_response.join()

        print(responses)
        for response in responses:
            if response["category"] == "wet":
                values[0] += response["confidence"]
            elif response["category"] == "plastic":
                values[1] += response["confidence"]
            elif response["category"] == "metal":
                values[2] += response["confidence"]
        if max(values) == 0:
            print("could not classify, switching to local model")
            print(local())
        else:
            max_index = values.index(max(values))
            if max_index == 0:
                print("wet") #insert function to move trash into wet waste bin here
            elif max_index == 1:
                print("plastic") #insert function to move trash into plastic bin here
            elif max_index == 2:
                print("metal") #insert function to move trash into metal bin here
            print(values)

