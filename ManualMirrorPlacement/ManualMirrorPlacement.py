import tkinter as tk
from tkinter import ttk
import math
from PIL import Image, ImageTk

from julia import Julia
from julia import Main

jl = Julia(compiled_modules=False) # TODO: try with True
Main.include("solutionEvaluation.jl")

def draw_line_at_angle(canvas, x, y, angle, linetag):
    """
    Draws a line on the canvas at a specific angle.
    
    :param canvas: The Tkinter Canvas on which to draw the line.
    :param x1: Starting x-coordinate of the line.
    :param y1: Starting y-coordinate of the line.
    :param length: Length of the line.
    :param angle: Angle in degrees (0 degrees is to the right).
    :param options: Other line options such as color, width, tags, etc.
    """
    # Convert angle to radians for the math functions
    length=14
    angle_rad = math.radians(angle)

    x1 = x - length/2 * math.cos(angle_rad)
    y1 = y + length/2 * math.sin(angle_rad)  # Subtract y because canvas Y-axis is inverted

    x2 = x + length/2 * math.cos(angle_rad)
    y2 = y - length/2 * math.sin(angle_rad)  # Subtract y because canvas Y-axis is inverted

    # Draw the line using the calculated points
    is_lamp = (linetag == "line0")
    color = ["blue", "purple"][is_lamp]
    canvas.create_line(x1, y1, x2, y2, fill=color, width=3, tags=linetag)


def on_canvas_click(event):
    # Get selected radio button index
    selected = radio_var.get()

    linetag="line"+str(selected)

    # Get the mouse click position (event.x, event.y)
    x, y = event.x, event.y

    canvas.delete(linetag)
    draw_line_at_angle(canvas, x, y, kutevi[selected], linetag)

#dimenzije ploce
dimenzije=600

kutevi=[0] * 10

def on_mouse_wheel(event):
    """Handle mouse wheel scrolling."""

    selected = radio_var.get()
    linetag="line"+str(selected)

    x1, y1, x2, y2 = canvas.coords(linetag)
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2

    # Scroll up or down depending on the direction of the scroll
    if event.delta > 0:  # Scroll up
        kutevi[selected]+=2
    else:  # Scroll down
        kutevi[selected]=(kutevi[selected]-2+360)%360

    canvas.delete(linetag)
    draw_line_at_angle(canvas, x, y, kutevi[selected], linetag)


def ispis():
    # No longer needs its own button
    filename="solution.txt"

    with open(filename, 'w') as file:
        # Write a single float number
        file.write(f"{3.33}\n")  # bezveze broj jer takav je tvoj format ucitavanja

        # Write up to 9 lines of 3 float numbers separated by space
        for i in range(9):
            linetag = f"line{i}"
            try:
                x1, y1, x2, y2 = canvas.coords(linetag)
                #x = (x1 + x2) / 2
                #y = (y1 + y2) / 2
                x, y, kut= x1/dimenzije*20, (dimenzije-y1)/dimenzije*20, kutevi[i]/360*2*math.pi
                numbers = f"{round(x, 6)} {round(y, 6)} {round(kut, 6)}"
                file.write(numbers + '\n')
            except ValueError:
                pass

def evaluacija():
    ispis()

    filename="solution.txt"
    result, result_img = Main.evaluate_and_draw(filename)
    formatted_result = f"{float(result):.2f}"
    result_var.set(f"Result: {formatted_result}")
    print(formatted_result)

    # Update the image
    new_image = Image.open(result_img)  # Replace with the path to your new image
    new_image = new_image.resize((dimenzije, dimenzije))  # Resize the image to fit the canvas
    global img  # Declare img as global to update it
    img = ImageTk.PhotoImage(new_image)
    canvas.itemconfig(image_on_canvas, image=img)


def move_selected_item(event):
    selected_index = radio_var.get()
    item = f"line{selected_index}"

    # Move the item based on the arrow key pressed
    if event.keysym == "Up":
        canvas.move(item, 0, -1)
    elif event.keysym == "Down":
        canvas.move(item, 0, 1)
    elif event.keysym == "Left":
        canvas.move(item, -1, 0)
    elif event.keysym == "Right":
        canvas.move(item, 1, 0)

# Create the main window
root = tk.Tk()
root.title("Mirror Placement")

# Configure grid layout
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)

# Load an image using PIL
image = Image.open("baza.png")  # Replace with the path to your image
image = image.resize((dimenzije, dimenzije))  # Resize the image to fit the canvas
img = ImageTk.PhotoImage(image)

# Create a canvas to display the image
canvas = tk.Canvas(root, width=dimenzije, height=dimenzije)
canvas.grid(row=0, column=0, padx=0, pady=0)
image_on_canvas = canvas.create_image(0, 0, anchor="nw", image=img)

# Bind mouse click events to the canvas (left click)
canvas.bind("<Button-1>", on_canvas_click)
canvas.bind("<MouseWheel>", on_mouse_wheel)

# Bind arrow key events to the canvas
canvas.bind_all("<Up>", move_selected_item)
canvas.bind_all("<Down>", move_selected_item)
canvas.bind_all("<Left>", move_selected_item)
canvas.bind_all("<Right>", move_selected_item)

# Create a variable to store the selected radio button index
radio_var = tk.IntVar(value=0)

# Create a frame on the right side for the radio buttons
radio_frame = ttk.Frame(root)
radio_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

# Create 9 radio buttons and place them in the frame
radio_button = tk.Radiobutton(radio_frame, text="Lampa", variable=radio_var, value=0)
radio_button.pack(anchor="w")

for i in range(1, 9):
    radio_button = tk.Radiobutton(radio_frame, text=f"Zrcalo {i}", variable=radio_var, value=i)
    radio_button.pack(anchor="w")

# Create a Label to display the result
result_var = tk.StringVar(value="Result: ")
result_label = ttk.Label(radio_frame, textvariable=result_var)
result_label.pack(pady=10)

# This button is no longer useful
# ispisi_button = ttk.Button(radio_frame, text="Ispisi", command=ispis)
# ispisi_button.pack(pady=10)  # Add some padding for aesthetics

evaluiraj_button = ttk.Button(radio_frame, text="Evaluiraj", command=evaluacija)
evaluiraj_button.pack(pady=10)  # Add some padding for aesthetics

# Run the main loop
root.mainloop()



