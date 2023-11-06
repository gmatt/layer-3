import logging
from io import BytesIO

import PIL
import pyautogui
import pyscreeze
import requests
from PIL.Image import Image

__PIL_TUPLE_VERSION = tuple(int(x) for x in PIL.__version__.split("."))
pyscreeze.PIL__version__ = __PIL_TUPLE_VERSION


VQA_URL = "http://localhost:8002/generate"
LOCALIZE_URL = "http://localhost:8001/localize"

# PROMPT = "I want to perform the following action: {action}. I currently see this screen. What should I click on next?"
PROMPT = "Question: I am trying to {action}. On what should I click next? Answer:"


def pil_to_bytesio(image: Image) -> BytesIO:
    temp = BytesIO()
    image.save(temp, format="png")
    temp.seek(0)
    return temp


def perform_action(action: str) -> None:
    screen_width, screen_height = pyautogui.size()
    logging.info("screen_width=%r screen_height=%r", screen_width, screen_height)

    logging.info("Creating screenshot")
    image: Image = pyautogui.screenshot()
    logging.info("Created screenshot")
    # Resize image, this is useful for retina resolution.
    image = image.resize((screen_width, screen_height))

    prompt = PROMPT.format(action=action)

    logging.info("Asking model %r...", prompt)
    response = requests.post(
        VQA_URL,
        files={"image": pil_to_bytesio(image)},
        data={"question": prompt},
    )
    response.raise_for_status()

    answer = response.json()["answer"]

    logging.info("Got answer %r", answer)

    logging.info("Asking to locate %r...", answer)

    response = requests.post(
        LOCALIZE_URL,
        files={"image": pil_to_bytesio(image)},
        data={"label": answer},
    )
    response.raise_for_status()

    coordinates = response.json()

    logging.info("Got coordinates %r", coordinates)

    pyautogui.click(coordinates["x"], coordinates["y"])


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    perform_action("click the execute button")
