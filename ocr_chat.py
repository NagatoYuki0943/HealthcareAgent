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
        req.ImageBase64 = get_file_content_as_base64(img)

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


# def ocr_chat(img, query, history: list, current_img: str):
#     print(f"{img = }")
#     print(f"{current_img = }")

#     # 有图片且图片不是之前的图片才使用ocr
#     if img != None and img != current_img:
#         print(f"use ocr")
#         ocr_result: str = ocr_detection(img, ocr_secret_id, ocr_secret_key)
#         txt = f"图片ocr检测结果:\n<ocr>\n{ocr_result}\n</ocr>\n question: {query}"
#         current_img = img
#     else:
#         txt = query
#     print(f"{txt = }")

#     url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-speed-128k?access_token=" + get_ernie_access_token(ernie_api_key, ernie_secret_key)

#     # 将历史记录转换为openai格式
#     prompts = convert_to_openai_history(history, txt)
#     print(f"{prompts = }")
#     # 注意message必须是奇数条
#     payload = json.dumps({"messages": prompts})
#     headers = {
#         'Content-Type': 'application/json'
#     }

#     if query == None and img == None:
#         return "", history, current_img
#     try:
#         res = requests.request("POST", url, headers=headers, data=payload).json()
#         response = res['result']
#         history.append([query, response])
#         print(f"{history = }")

#         return "", history, current_img
#     except Exception as e:
#         return e, history, current_img


# def main() -> None:
#     block = gr.Blocks()
#     with block as demo:
#         state_session_id = gr.State(0)

#         with gr.Row(equal_height=True):
#             with gr.Column(scale=15):
#                 gr.Markdown("""<h1><center>Healthcare Agent</center></h1>""")
#             # gr.Image(value=LOGO_PATH, scale=1, min_width=10,show_label=False, show_download_button=False)


#         # 化验报告分析页面
#         with gr.Tab("化验报告分析"):
#             # 用来存放ocr图片路径，防止重复使用ocr
#             current_img = gr.State("")

#             gr.Markdown("""<h1><center>报告分析 Healthcare Textract</center></h1>""")
#             with gr.Row():

#                 img_chatbot = gr.Chatbot(height=450, show_copy_button=True)
#                 img_show = gr.Image(sources=["upload", "webcam", "clipboard"], type="filepath", label="输入的化验报告图片", height=450)

#             with gr.Row():
#                 question = gr.Textbox(label="Prompt/问题", scale=2)
#                 # img_intput = gr.UploadButton('📁', elem_id='upload', file_types=['image'], scale=0)
#                 # print(img_intput.name)
#                 subbt = gr.Button(value="Chat", variant="primary", scale=0)
#                 # 创建一个清除按钮，用于清除聊天机器人组件的内容。
#                 clear = gr.ClearButton(components=[img_chatbot, img_show, current_img], value="Clear", variant="stop", scale=0)

#         subbt.click(ocr_chat, inputs=[img_show, question, img_chatbot, current_img], outputs=[question, img_chatbot, current_img])
#         question.submit(ocr_chat, inputs=[img_show, question, img_chatbot, current_img], outputs=[question, img_chatbot, current_img])


#         # 智能问答页面
#         with gr.Tab("医疗智能问答"):
#             ...

#         gr.Markdown("""
#         ### 内容由 AI 大模型生成，不构成专业医疗意见或诊断。
#         """)

#     # threads to consume the request
#     gr.close_all()

#     # 设置队列启动
#     demo.queue(
#         max_size = None,                # If None, the queue size will be unlimited.
#         default_concurrency_limit = 40  # 最大并发限制
#     )

#     # demo.launch(server_name = "127.0.0.1", server_port = 7860, share = True, max_threads = 40)
#     demo.launch(max_threads = 40)


# if __name__ == "__main__":
#     main()