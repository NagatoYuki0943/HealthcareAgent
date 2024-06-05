import gradio as gr
import cv2
import pandas as pd
import base64
import urllib
import json
import requests
import os
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.ocr.v20181119 import ocr_client, models


# https://blog.csdn.net/weixin_30347335/article/details/95849160

def get_file_content_as_base64(path, urlencoded=False):
    """
    获取文件base64编码
    :param path: 文件路径
    :param urlencoded: 是否对结果进行urlencoded
    :return: base64编码信息
    """
    with open(path, "rb") as f:
        # print(f.read())
        content = str(base64.b64encode(f.read()), "utf-8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content


def ocr_detection(img, secret_id, secret_key):

    try:

        cred = credential.Credential(secret_id, secret_key)
        # 实例化一个http选项，可选的，没有特殊需求可以跳过
        httpProfile = HttpProfile()
        httpProfile.endpoint = "ocr.tencentcloudapi.com"

        # 实例化一个client选项，可选的，没有特殊需求可以跳过
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        # 实例化要请求产品的client对象,clientProfile是可选的
        client = ocr_client.OcrClient(cred, "ap-guangzhou", clientProfile)

        # 实例化一个请求对象,每个接口都会对应一个request对象
        req = models.RecognizeTableAccurateOCRRequest()
        # params = {
        #     "ImageUrl": "ImageUrl"
        # }
        # req.from_json_string(json.dumps(params))
        req.ImageBase64 = get_file_content_as_base64(img.name)

        # 返回的resp是一个RecognizeTableAccurateOCRResponse的实例，与请求对象对应
        resp = client.RecognizeTableAccurateOCR(req)
        # 输出json格式的字符串回包 输出结果含义： https://cloud.tencent.com/document/api/866/33527#TableInfo
        data = resp.to_json_string()
        data_json = pd.read_json(data)

        rowIndex = []
        colIndex = []
        content = []

        for data_tabled in data_json['TableDetections']:
            for item in data_tabled['Cells']:
                rowIndex.append(item['RowTl'])
                colIndex.append(item['ColTl'])
                content.append(item['Text'] if '~' not in item['Text'] else item['Text'].replace("~", '-'))

        return str(content)

    except TencentCloudSDKException as err:
        print(err)


def get_ernie_access_token(ernie_api_key, ernie_secret_key):
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": ernie_api_key, "client_secret": ernie_secret_key}
    return str(requests.post(url, params=params).json().get("access_token"))


# def ocr_chat(img, query, history:list):
#     txt = ocr_detection(img, ocr_secret_id, ocr_secret_key) + "," + query if img != None else query
#     show_img = cv2.imread(img.name) if img!= None else None


#     url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-speed-128k?access_token=" + get_ernie_access_token(ernie_api_key, ernie_secret_key)
#     # 注意message必须是奇数条
#     payload = json.dumps({
#     "messages": [
#         {
#             "role": "user",
#             "content": txt,
#         }
#     ]
#     })
#     headers = {
#         'Content-Type': 'application/json'
#     }

#     if query == None and img == None:
#         return "", show_img, history, None
#     try:
#         res = requests.request("POST", url, headers=headers, data=payload).json()
#         response = res['result']
#         history.append((query, response))

#         return "", show_img, history, None
#     except Exception as e:
#         return e, show_img, history, None


# def main():
#     # 创建一个 Web 界面
#     block = gr.Blocks()
#     with block as demo:

#         # 化验报告分析页面
#         with gr.Tab("化验报告分析"):
#             gr.Markdown("""<h1><center>报告分析 Healthcare Textract</center></h1>
#                             """)
#             with gr.Row():

#                 img_chatbot = gr.Chatbot(height=450, show_copy_button=True)
#                 img_show = gr.Image(label="输入的化验报告图片", height=450)

#             with gr.Row():
#                 question = gr.Textbox(label="Prompt/问题", scale=2)
#                 img_intput = gr.UploadButton('📁', elem_id='upload', file_types=['image'], scale=0)
#                 # print(img_intput.name)
#                 subbt = gr.Button(value="Chat", scale=0)


#         subbt.click(ocr_chat, inputs=[img_intput, question, img_chatbot], outputs=[question, img_show, img_chatbot, img_intput])
#         question.submit(ocr_chat, inputs=[img_intput, question, img_chatbot], outputs=[question, img_show, img_chatbot, img_intput])

#         # 智能问答页面
#         with gr.Tab("智能问答"):
#             chatbot2 = gr.Chatbot(height=450, show_copy_button=True)

#     gr.close_all()
#     # 直接启动
#     demo.launch()


# if __name__ == "__main__":
#     main()