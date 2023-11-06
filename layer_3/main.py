import PySimpleGUI as sg

sg.set_options(font=("Arial", 20))

# sg.theme("Material2")  # Add a touch of color
# All the stuff inside your window.
layout = [
    [sg.Text("What should I do?"), sg.InputText(), sg.Button("Perform")],
]

# Create the Window
window = sg.Window("Layer 3 Demo App", layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    print("You entered ", values[0])

window.close()
