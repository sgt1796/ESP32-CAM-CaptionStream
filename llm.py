"""Backend LLM Class modified for LLM captioning"""
import base64
from openai import OpenAI
import os
import tempfile
from io import BytesIO
from PIL import Image
from typing import Callable, Any, Optional, Tuple, List
from dotenv import load_dotenv
from fastapi import UploadFile
from google import genai

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

#from api import *

class LLM:

    def __init__(self, model: str = "openai") -> None:
        if model == "openai":
            self.image_caption = self.__init_openai_image_caption__
        elif model == "gemini":
            # those 2 use the same function __init_openai_image_recognize__
            self.image_recognize = self.__init_gemini_image_recognize__
            self.image_caption = self.__init_gemini_image_recognize__
        else:
            raise ValueError("Model not found")
        
    def caption_with_objects(
        self,
        imgbase64: str,
        objects: List[Tuple[str, float]] | None = None,
        resize: Optional[Tuple[int, int]] = None,
    ) -> str:
        """
        Wrapper that builds a prompt from detector output (e.g. YOLO)
        and calls the usual image_caption().
        Args
        ----
        imgbase64 : base-64 JPEG
        objects   : list of (label, confidence) pairs *already in human words*
        resize    : optional (w,h)  – passed straight through
        """
        if objects:
            # "person 0.93, dog 0.71"
            hint = ", ".join(f"{lbl} {conf:.2f}" for lbl, conf in objects)
            prompt = (
                "You are an image-captioning assistant.\n"
                f"The detector sees: {hint}.\n"
                "Write concise English sentence describing the scene. "
                "Only mention objects if you also sees it."
            )
        else:
            prompt = "You are an image-captioning assistant.\nDescribe this image."

        return self.image_caption(
            imgbase64=imgbase64,
            prompt=prompt,
            resize=resize,
        )
    
    def __process_image_input__(
        self, 
        imgbase64: Optional[str] = None, 
        image_file: Optional[UploadFile] = None, 
        resize: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """
        Given a base64 string or an uploaded image file, load and optionally resize the image.
        """
        if imgbase64:
            img_data = base64.b64decode(imgbase64)
            image = Image.open(BytesIO(img_data))
        elif image_file:
            contents = image_file.file.read()
            image = Image.open(BytesIO(contents))
        else:
            raise ValueError("Either imgbase64 or image_file must be provided.")

        if resize is not None:
            image = image.resize(resize)
        return image

    def __init_openai_image_caption__(
        self, 
        imgbase64: Optional[str] = None, 
        image_file: Optional[UploadFile] = None, 
        prompt: str = "Describe this image.",
        resize: Optional[Tuple[int, int]] = None
    ) -> str:
        """
        Generate a zero-shot image caption using OpenAI GPT-4o Vision API.
        Accepts either base64-encoded image (imgbase64) or an uploaded image file (image_file).
        Optionally resizes the image if 'resize' is specified (tuple of (w, h)).
        Returns: caption string
        """
        api_key = str(OPENAI_API_KEY)
        client = OpenAI(
            api_key=api_key,
        )

        # Load and optionally resize the image
        image = self.__process_image_input__(imgbase64, image_file, resize)

        # Save to a temporary file (OpenAI vision API expects a file-like object)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp_file, format="JPEG")
        temp_file.close()

        # Re-encode to base64 for API
        with open(temp_file.name, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode('utf-8')

        with open(temp_file.name, "rb") as image_file_obj:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert image captioning assistant."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]}
                ],
                max_tokens=64,
            )
        # Return the generated caption
        return completion.choices[0].message.content.strip()

    def __init_gemini_image_recognize__(self, 
                                        imgbase64: Optional[str] = None, 
                                        image_file: Optional[UploadFile] = None, 
                                        task: Optional[str] = "caption",
                                        resize: Optional[Tuple[int, int]] = None
                                        ) -> str:
        """Recognize image with Gemini."""
        image = self.__process_image_input__(imgbase64=imgbase64, image_file=image_file, resize=resize)

        client = genai.Client(api_key=GEMINI_API_KEY)
        if task == "配诗":
            prompt = "识别图片中的内容，给图片配一首中文诗，要求生动有趣，朗朗上口。只需要回复诗句本身，不需要其他内容。"
            image = self.__process_image_input__(imgbase64=imgbase64, resize=(640, 640)) # default setting

        elif task == "caption" or task == "描述":
            prompt = "识别图片中的内容，给图片配一个简短的描述。只需要回复描述本身，不需要其他内容。"
        else:
            raise ValueError("Task not found")

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[image, prompt]
        )
        return response.text