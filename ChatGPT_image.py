import os
import torch
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import json
import comfy.utils
import aiohttp
import asyncio
import base64
import numpy as np
import re
import mimetypes
from typing import List, Union


# Load configuration file
def load_config():
    try:
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')
        with open(config_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("Config file config.json not found")
    except json.JSONDecodeError:
        print("Invalid config.json format")
    except Exception as e:
        print(f"Error reading config file: {e}")
    return {}


# Save configuration file
def save_configuration(config):
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')
    try:
        with open(config_path, 'w') as file:
            json.dump(config, file, indent=4)
    except Exception as e:
        print(f"Error saving config file: {e}")


# Create error image
def create_error_image():
    image = Image.new('RGB', (512, 512), color='white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((200, 250), 'ERROR', fill='red', font=font)
    return pil2tensor(image)


# Convert image to base64
def image_to_base64_str(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# Convert file to base64 and get MIME type
def file_to_base64_with_mime(file_path):
    try:
        with open(file_path, "rb") as file:
            content = file.read()
            encoded = base64.b64encode(content).decode('utf-8')
            mime_type, _ = mimetypes.guess_type(file_path)
            return encoded, mime_type or "application/octet-stream"
    except Exception as e:
        print(f"Error encoding file: {e}")
        return None, None


# Extract image URLs from response text
def extract_image_urls_from_response(response_text):
    patterns = [
        (r'!\[.*?\]\((.*?)\)', "Markdown格式"),
        (r'https?://\S+\.(?:jpg|jpeg|png|gif|webp)', "图片链接"),
        (r'https?://\S+', "通用链接")
    ]
    for pattern, _ in patterns:
        matches = re.findall(pattern, response_text)
        if matches:
            return matches
    return []


# Download image and convert to tensor
def download_and_convert_image(url, timeout=30):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://comfyui.com/'
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('image/'):
            print(f"Warning: URL may not be an image. Content-Type: {content_type}")
            return None
        return pil2tensor(Image.open(BytesIO(response.content)))
    except requests.exceptions.Timeout:
        print(f"Image download timeout: {url} (timeout: {timeout}s)")
    except requests.exceptions.SSLError as e:
        print(f"SSL error: {url}: {e}")
    except requests.exceptions.ConnectionError:
        print(f"Connection error: {url}")
    except requests.exceptions.RequestException as e:
        print(f"Request error: {url}: {e}")
    except Exception as e:
        print(f"Error downloading image: {url}: {e}")
    return None


# Convert PIL image to tensor
def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(image, list):
        if not image:
            return torch.empty(0)
        tensors = []
        for img in image:
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(img_array)[None,]
            tensors.append(tensor)
        if len(tensors) == 1:
            return tensors[0]
        shapes = [t.shape[1:3] for t in tensors]
        if all(shape == shapes[0] for shape in shapes):
            return torch.cat(tensors, dim=0)
        max_h = max(t.shape[1] for t in tensors)
        max_w = max(t.shape[2] for t in tensors)
        padded_tensors = []
        for t in tensors:
            h, w = t.shape[1:3]
            if h == max_h and w == max_w:
                padded_tensors.append(t)
            else:
                padded = torch.zeros((1, max_h, max_w, 3), dtype=t.dtype)
                padded[0, :h, :w, :] = t[0, :h, :w, :]
                padded_tensors.append(padded)
        return torch.cat(padded_tensors, dim=0)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(img_array)[None,]


# Convert tensor to PIL image
def tensor2pil(image: torch.Tensor) -> List[Image.Image]:
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        images = []
        for i in range(batch_count):
            images.extend(tensor2pil(image[i]))
        return images
    numpy_image = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return [Image.fromarray(numpy_image)]


class ChatGPT_Image:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "font_size": 24}),
                "base_url": ("STRING", {"default": "http://www.deeplpro.com"}),
                "api_key": ("STRING", {"default": ""})
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",)
            }
        }
    
    NAME = "ChatGPT_Image"
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_inputs"
    CATEGORY = "image"

    def __init__(self):
        config = load_config()
        self.base_url = config.get('base_url', 'http://www.deeplpro.com')
        self.api_key = config.get('api_key', '')
        self.timeout = 800
        self.image_download_timeout = 600
        self.api_endpoint = f"{self.base_url}/v1/chat/completions"

    def get_request_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    async def stream_api_response(self, payload, progress_bar):
        full_response = ""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        self.api_endpoint,
                        headers=self.get_request_headers(),
                        json=payload,
                        timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API error {response.status}: {error_text}")
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data)
                                if 'choices' in chunk and chunk['choices']:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        content = delta['content']
                                        full_response += content
                                        current_progress = min(40, 20 + len(full_response) // 50)
                                        progress_bar.update_absolute(current_progress)
                                        print(f"Generating response... Progress: {current_progress}%")
                            except json.JSONDecodeError:
                                continue
            return full_response
        except asyncio.TimeoutError:
            raise TimeoutError(f"API request timeout, timeout duration: {self.timeout} seconds")
        except Exception as e:
            raise Exception(f"Streaming response error: {e}")

    def process_inputs(self, prompt, image1, image2, base_url="", api_key=""):
        config = load_config()
        if base_url.strip():
            self.base_url = base_url
            config['base_url'] = base_url
            self.api_endpoint = f"{self.base_url}/v1/chat/completions"
        if api_key.strip():
            self.api_key = api_key
            config['api_key'] = api_key
        save_configuration(config)
        try:
            if not self.api_key:
                print("API key not found in config.json")
                return (create_error_image(),)
            progress_bar = comfy.utils.ProgressBar(100)
            progress_bar.update_absolute(10)
            content = [{"type": "text", "text": prompt}]
            for img in [image1, image2]:
                if img is not None:
                    pil_image = tensor2pil(img)[0]
                    image_base64 = image_to_base64_str(pil_image)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                    })
            messages = [{"role": "user", "content": content}]
            payload = {
                "model": "gpt-4o-image",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 4096,
                "stream": True
            }
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response_text = loop.run_until_complete(self.stream_api_response(payload, progress_bar))
            except Exception as e:
                print(f"API call error: {e}")
                return (create_error_image(),)
            loop.close()
            image_urls = extract_image_urls_from_response(response_text)
            if image_urls:
                img_tensors = []
                successful_downloads = 0
                total_images = len(image_urls)
                for i, url in enumerate(image_urls):
                    print(f"Attempting to download image {i + 1}/{total_images} from: {url}")
                    progress = 40 + (i * 40 // total_images)
                    progress_bar.update_absolute(min(80, progress))
                    img_tensor = download_and_convert_image(url, self.image_download_timeout)
                    if img_tensor is not None:
                        img_tensors.append(img_tensor)
                        successful_downloads += 1
                print(f"Successfully downloaded {successful_downloads} of {total_images} images")
                if img_tensors:
                    combined_tensor = torch.cat(img_tensors, dim=0)
                    progress_bar.update_absolute(100)
                    return (combined_tensor,)
            progress_bar.update_absolute(100)
            return (image1,)
        except Exception as e:
            print(f"Error processing input: {e}")
            return (create_error_image(),)


WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "ChatGPT_Image": ChatGPT_Image
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatGPT_Image": "ChatGPT_Image"
}
    