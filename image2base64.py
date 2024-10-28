import base64
import imghdr
from PIL import Image
import io


def convert_image_to_base64(image_path: str, target_format=None):
    """
    转换任意格式图片为base64，可选择转换为指定格式
    """
    try:
        # 如果需要转换格式
        if target_format:
            with Image.open(image_path) as img:
                # 创建内存缓冲区
                buffer = io.BytesIO()
                # 保存为目标格式
                img.save(buffer, format=target_format)
                # 获取字节数据
                image_data = buffer.getvalue()
        else:
            # 直接读取原格式
            with open(image_path, "rb") as f:
                image_data = f.read()

        # base64编码
        base64_data = base64.b64encode(image_data).decode("utf-8")

        # 获取实际格式
        img_format = target_format or imghdr.what(image_path)
        prefix = f"data:image/{img_format.lower()};base64,"

        return prefix + base64_data

    except Exception as e:
        print(f"转换错误: {e}")
        return None


def convert_base64_to_image(base64_str: str, output_path: str):
    try:
        # 移除base64字符串前缀
        if "data:image" in base64_str:
            base64_data = base64_str.split(",")[1]
        else:
            base64_data = base64_str

        # 解码并保存图片
        image_data = base64.b64decode(base64_data)
        with open(output_path, "wb") as f:
            f.write(image_data)
        return True
    except Exception as e:
        print(f"解码错误: {e}")
        return False


if __name__ == '__main__':
    image_path = "image1.jpg"
    base64_str = convert_image_to_base64(image_path)
    print(base64_str)
    convert_base64_to_image(base64_str, "image2.jpg")
