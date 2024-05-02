config_dict=dict(
    access_config = dict(
        api_key="", # add your api_key here
        api_secret="", # add your api_secret here
        client_credentials_address="https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials",
    ),
    model_config = dict(
        model_dict={
        "ERNIE-Lite-8K":'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-lite-8k',
        "ERNIE-4.0-8K":'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro',
        })
)