import tkinter as tk
from tkinter import ttk
import math
from PIL import Image, ImageTk


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
    length=18
    angle_rad = math.radians(angle)

    
    x1 = x - length/2 * math.cos(angle_rad)
    y1 = y + length/2 * math.sin(angle_rad)  # Subtract y because canvas Y-axis is inverted
    
    x2 = x + length/2 * math.cos(angle_rad)
    y2 = y - length/2 * math.sin(angle_rad)  # Subtract y because canvas Y-axis is inverted

    # Draw the line using the calculated points
    canvas.create_line(x1, y1, x2, y2, fill="blue", width=3, tags=linetag)


def on_canvas_click(event):
    # Get selected radio button index
    selected = radio_var.get()

    linetag="line"+str(selected)
    
    # Get the mouse click position (event.x, event.y)
    x, y = event.x, event.y
    poz_x[selected]=x
    poz_y[selected]=y

    canvas.delete(linetag)
    draw_line_at_angle( canvas , x , y  , kutevi[selected] , linetag )

#dimenzije ploce
dimenzije=600

poz_x=[0] * 10
poz_y=[0] * 10
kutevi=[0] * 10

def on_mouse_wheel(event):
    """Handle mouse wheel scrolling."""

    selected = radio_var.get()
    linetag="line"+str(selected)

    x=poz_x[selected]
    y=poz_y[selected]
    
    # Scroll up or down depending on the direction of the scroll
    if event.delta > 0:  # Scroll up
        kutevi[selected]+=2

        canvas.delete(linetag)
        draw_line_at_angle( canvas , x , y  , kutevi[selected] , linetag )
        
    else:  # Scroll down
        kutevi[selected]=(kutevi[selected]-2+360)%360
        
        canvas.delete(linetag)
        draw_line_at_angle( canvas , x , y  , kutevi[selected] , linetag )
        

def ispis():
    filename="solution.txt"
    
    with open(filename, 'w') as file:
        # Write a single float number
        file.write(f"{3.33}\n")  # bezveze broj jer takav je tvoj format ucitavanja

        # Write 9 lines of 3 float numbers separated by space
        for i in range(9):
            x,y,kut=poz_x[i] / dimenzije*20,(dimenzije-poz_y[i]) / dimenzije*20,kutevi[i]/360*2*math.pi
            numbers = f"{round(x, 6)} {round(y, 6)} {round(kut, 6)}"
            file.write(numbers + '\n')
        

# Create the main window
root = tk.Tk()
root.title("Draw Lines Based on Events")

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
canvas.create_image(0, 0, anchor="nw", image=img)

# Bind mouse click events to the canvas (left click)
canvas.bind("<Button-1>", on_canvas_click)
canvas.bind("<MouseWheel>", on_mouse_wheel)

# Create a variable to store the selected radio button index
radio_var = tk.IntVar(value=0)

# Create a frame on the right side for the radio buttons
radio_frame = ttk.Frame(root)
radio_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

# Create 9 radio buttons and place them in the frame
radio_button = tk.Radiobutton(radio_frame, text=f"Lampa", variable=radio_var, value=0)
radio_button.pack(anchor="w")

for i in range(1, 9):
    radio_button = tk.Radiobutton(radio_frame, text=f"Zrcalo {i}", variable=radio_var, value=i)
    radio_button.pack(anchor="w")


ispisi_button = ttk.Button(radio_frame, text="Ispisi", command=ispis)
ispisi_button.pack(pady=10)  # Add some padding for aesthetics

# Run the main loop
root.mainloop()



