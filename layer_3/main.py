import logging

import PySimpleGUI as sg

from layer_3.logic import perform_action


def main():
    sg.set_options(font=("Arial", 20))

    # sg.theme("Material2")  # Add a touch of color
    # All the stuff inside your window.
    layout = [
        [sg.Text("What should I do?")],
        [sg.InputText(), sg.Button("Perform")],
    ]

    # Create the Window
    window = sg.Window("VL-LLM Automation Demo App", layout)
    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        perform_action(values[0])

    window.close()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
