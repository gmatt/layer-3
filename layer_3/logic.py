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
# PROMPT = "Question: I am trying to {action}. Where should I click next? Answer:"
PROMPT = "Task: {action}, Choices: click <button name>, press <keyboard key>, type <text>, done. Answer:"


def pil_to_bytesio(image: Image) -> BytesIO:
    temp = BytesIO()
    image.save(temp, format="png")
    temp.seek(0)
    return temp


def perform_action(action: str) -> None:
    try:
        screen_width, screen_height = pyautogui.size()
        logging.info("screen_width=%r screen_height=%r", screen_width, screen_height)

        for i in range(5):
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

            answer: str = response.json()["answer"]

            logging.info("Got answer %r", answer)

            if answer.strip().lower().startswith("click"):
                answer = answer.strip().lower().replace("click", "").strip()
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
            elif answer.strip().lower().startswith("type"):
                answer = answer.strip().lower().replace("type", "").strip()
                logging.info("Typing %r", answer)
                pyautogui.write(answer, interval=0.25)
            elif answer.strip().lower().startswith("press"):
                answer = answer.strip().lower().replace("press", "").strip()
                logging.info("pressing %r", answer)
            elif answer.strip().lower().startswith("done"):
                logging.info("Received done", answer)
                return
            else:
                logging.info("Unknown result: %r", answer)

    except pyautogui.FailSafeException:
        return


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    perform_action("click the execute button")
