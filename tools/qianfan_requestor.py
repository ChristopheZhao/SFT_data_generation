import requests  
import json
from config import config_dict
from tools.cfg_wapper import load_config
  
class BaiduAIChat:  
    def __init__(self, api_key, secret_key,api_address,client_credentials_address):  
        self.api_key = api_key  
        self.secret_key = secret_key  
        self.api_address = api_address
        self.client_credentials_address = client_credentials_address
        self.access_token = self.get_access_token()  
        self.req_url = f"{self.api_address}?access_token={self.access_token}"
  
    def get_access_token(self):  
        url = (  
            f"{self.client_credentials_address}"  
            f"&client_id={self.api_key}&client_secret={self.secret_key}"  
        )  
        payload = json.dumps("")
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json().get("access_token")
    

    def send_message(self, message):  

        payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": f"{message}"
            }
                    ]
         })
        headers = {
            'Content-Type': 'application/json'
        }
        
        response = requests.request("POST", self.req_url, headers=headers, data=payload)


        return response.json()
    
# initial qianfan_requestor use the following code
def init_qianfan_requestor(config_dict):
    config = load_config(config_dict)

    access_config = config.access_config

    api_key = access_config.api_key
    api_secret = access_config.api_secret
    client_credentials_address = access_config.client_credentials_address

    model_config = config.model_config
    api_model_address = model_config.model_dict.getitem('ERNIE-4.0-8K')
    baidu_ai_chat = BaiduAIChat(api_key, api_secret,api_model_address,client_credentials_address)
    return baidu_ai_chat

  
# 使用示例  
if __name__ == '__main__':  

    config = load_config(config_dict)

    access_config = config.access_config

    api_key = access_config.api_key
    api_secret = access_config.api_secret
    client_credentials_address = access_config.client_credentials_address


    model_config = config.model_config
    api_model_address = model_config.model_dict.getitem('ERNIE-4.0-8K')
    baidu_ai_chat = BaiduAIChat(api_key, api_secret,api_model_address,client_credentials_address)  
    
    res = baidu_ai_chat.send_message("你好")

    print(res)