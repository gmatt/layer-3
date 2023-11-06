from io import BytesIO
from typing import Optional

import numpy as np
import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile
from lavis.common.gradcam import getAttMap
from lavis.models import load_model_and_preprocess
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
from matplotlib import pyplot as plt
from pydantic import BaseModel

app = FastAPI()


class ResponseModel(BaseModel):
    x: float
    y: float


model: Optional[torch.nn.Module]
vis_processors: Optional[dict]
text_processors: Optional[dict]
# TODO Enable GPU.
device = "cpu"


@app.on_event("startup")
async def startup_event():
    global model, vis_processors, text_processors

    model, vis_processors, text_processors = load_model_and_preprocess(
        "blip_image_text_matching",
        "large",
        device=device,
        is_eval=True,
    )


@app.post("/localize")
async def localize(
    label: str = Form(...),
    image: UploadFile = File(...),
) -> ResponseModel:
    raw_image = Image.open(BytesIO(await image.read())).convert("RGB")
    # plt.imshow(raw_image)
    # plt.show()

    dst_w = 720
    w, h = raw_image.size
    scaling_factor = dst_w / w

    resized_img = raw_image.resize((int(w * scaling_factor), int(h * scaling_factor)))
    norm_img = np.float32(resized_img) / 255
    raw_image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    txt = text_processors["eval"](label)
    txt_tokens = model.tokenizer(txt, return_tensors="pt").to(device)
    gradcam, _ = compute_gradcam(model, raw_image, txt, txt_tokens, block_num=7)

    avg_gradcam_2 = getAttMap(
        norm_img,
        gradcam[0][1].cpu().numpy(),
        blur=True,
        overlap=False,
    )

    coordinates = np.array(
        np.unravel_index(avg_gradcam_2.argmax(), avg_gradcam_2.shape)
    )

    avg_gradcam = getAttMap(norm_img, gradcam[0][1].cpu().numpy(), blur=True)
    plt.imshow(avg_gradcam)
    plt.plot(coordinates[1], coordinates[0], marker="+", markersize=20, c="red")
    plt.show()

    return ResponseModel(
        x=coordinates[1] * scaling_factor,
        y=w - (coordinates[0] * scaling_factor),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
