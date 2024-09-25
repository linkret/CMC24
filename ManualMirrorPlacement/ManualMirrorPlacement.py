import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import math
import sys
import time
import subprocess
import os
from PIL import Image, ImageTk

from julia import Julia
from julia import Main

old_image = None
curr_score = 0.0

start_time = time.time()

daemon_mode = '-d' in sys.argv

#dimenzije ploce
dimenzije=1000
debljina_crte=4
kutevi=[0] * 9

if daemon_mode:
    result = subprocess.run(
        ["julia", "-e", "using DaemonMode; runargs()", "solutionEvaluation.jl"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Error running Daemon Julia script: {result.stderr}")
    else:
        # print(result)
        pass

    if result.stderr.startswith("Error, cannot connect with server."):
        # Start the persistant Julia Daemon process
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        
        # The following command starts a persistant Julia REPL server in the background
        # Need to first execute ```using Pkg Pkg.add("DaemonMode")``` once, to globaly download the DaemonMode package

        process = subprocess.Popen(
            ["julia", "-e", "using DaemonMode; serve(3000, true; print_stack=true);"],
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        print(f"Started Julia Daemon with PID: {process.pid}")

        # Try to run the script again
        result = subprocess.run(
            ["julia", "-e", "using DaemonMode; runargs()", "solutionEvaluation.jl"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Error running Daemon Julia script: {result.stderr}")
        else:
            # print(result.stdout)
            pass
else:
    jl = Julia(compiled_modules=False)
    Main.include("solutionEvaluation.jl")

end_time = time.time()
print(f"Julia initialization time: {end_time - start_time:.2f} seconds")


def upis_koordinata():
    selected = radio_var.get()

    text1.delete(0, tk.END)  # Clear the existing text
    text2.delete(0, tk.END)  # Clear the existing text
    text3.delete(0, tk.END)  # Clear the existing text

    try:
        linetag="line"+str(selected)

        x1, y1, x2, y2 = canvas.coords(linetag)
        xx = (x1 + x2) / 2
        yy = (y1 + y2) / 2

        x, y, kut= xx/dimenzije*20, (dimenzije-yy)/dimenzije*20, kutevi[i]/360*2*math.pi #ovak mora bit jer autizam sastavljaca
        
        # Insert new text
        text1.insert(0, str(x))
        text2.insert(0, str(y))
        text3.insert(0, str(kutevi[selected]))
    except ValueError:
        pass

def mapa_boja():
    boje={}
    boje["line0"]="purple"
    for i in range(1,9):
        boje["line"+str(i)]="blue"

    boje[f"line{radio_var.get()}"]="yellow"
    
    return boje

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
    length=(0.5/20)*dimenzije
    angle_rad = math.radians(angle)

    x1 = x - length/2 * math.cos(angle_rad)
    y1 = y + length/2 * math.sin(angle_rad)  # Subtract y because canvas Y-axis is inverted

    x2 = x + length/2 * math.cos(angle_rad)
    y2 = y - length/2 * math.sin(angle_rad)  # Subtract y because canvas Y-axis is inverted

    # Draw the line using the calculated points
    canvas.delete(linetag)
    canvas.create_line(x1, y1, x2, y2, fill=mapa_boja()[linetag], width=debljina_crte, tags=linetag)
    upis_koordinata()

def on_enter(event):

    selected=radio_var.get()

    linetag="line"+str(selected)
    
    x=float(text1.get())
    y=float( text2.get() )
    kutevi[selected]=float(text3.get())

    draw_line_at_angle(canvas, x, y, kutevi[selected], linetag)

def on_canvas_click(event):
    # Get selected radio button index
    selected = radio_var.get()

    linetag="line"+str(selected)

    # Get the mouse click position (event.x, event.y)
    x, y = event.x, event.y

    draw_line_at_angle(canvas, x, y, kutevi[selected], linetag)
    upis_koordinata()

def on_mouse_wheel(event):
    """Handle mouse wheel scrolling."""

    selected = radio_var.get()
    linetag="line"+str(selected)

    x1, y1, x2, y2 = canvas.coords(linetag)
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2

    # Scroll up or down depending on the direction of the scroll
    if event.delta > 0:  # Scroll up
        kutevi[selected]=(kutevi[selected]+1+360)%360
    else:  # Scroll down
        kutevi[selected]=(kutevi[selected]-1+360)%360

    draw_line_at_angle(canvas, x, y, kutevi[selected], linetag)
    upis_koordinata()

def on_select():#it doesnt have event
    selected = radio_var.get()
    linetag="line"+str(selected)
    
    for i in range(9):
        linetag = f"line{i}"
        try:
            x1, y1, x2, y2 = canvas.coords(linetag)
            canvas.delete(linetag)
            canvas.create_line(x1, y1, x2, y2, fill=mapa_boja()[linetag], width=debljina_crte, tags=linetag)
        except ValueError:
            pass
    upis_koordinata()

def ispis(filename="solution.txt"):
    with open(filename, 'w') as file:
        file.write(f"{curr_score}\n") # will be wrong on evaluacija() calls, but doesn't matter

        # Write up to 9 lines of 3 float numbers separated by space
        for i in range(9):
            linetag = f"line{i}"
            try:
                x1, y1, x2, y2 = canvas.coords(linetag)
                x, y, kut= x1/dimenzije*20, (dimenzije-y1)/dimenzije*20, kutevi[i]/360*2*math.pi #ovak mora bit jer autizam sastavljaca
                numbers = f"{round(x, 6)} {round(y, 6)} {round(kut, 6)}"
                file.write(numbers + '\n')
            except ValueError:
                pass

def save_file():
    filename = filedialog.asksaveasfilename(
        title="Save file",
        defaultextension=".txt",
        filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
    )

    if filename:
        ispis(filename)

def evaluacija():
    global old_image, curr_score
    ispis()
    filename="solution.txt"

    if daemon_mode:
        result = subprocess.run(
            ["julia", "-e", "using DaemonMode; runexpr(\"evaluate_and_draw(\\\"solution.txt\\\")\")"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Error running Julia script: {result.stderr}")
            return
        
        # print(result)

        output_lines = result.stdout.strip().split('\n')
        if len(output_lines) < 2 or output_lines[-1] == "0":
            print("Unexpected output from Julia script: ", result.stdout)
            return
        else:
            result_value = float(output_lines[-2])
            result_img = output_lines[-1]
    else:
        result_value, result_img = Main.evaluate_and_draw(filename)

    formatted_result = f"{result_value:.2f}"
    result_var.set(f"Result: {formatted_result}")
    curr_score = result_value
    print(formatted_result)

    if result_img is None:
        return

    # Update the image
    new_image = Image.open(result_img)
    new_image = new_image.resize((dimenzije, dimenzije))  # Resize the image to fit the canvas
    global img  # Declare img as global to update it
    img = ImageTk.PhotoImage(new_image)
    canvas.itemconfig(image_on_canvas, image=img)
    if old_image is not None and os.path.exists(old_image):
        os.remove(old_image)
    old_image = result_img

def load_file():
    filename = filedialog.askopenfilename(
        title="Select a file",
        filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
    )

    with open(filename, 'r') as file:

        initial_number = float(file.readline().strip())

        for i in range(9):
            canvas.delete(f"line{i}")
        
        i = 0
        while True:
            linetag = f"line{i}"
            line = file.readline().strip()
            if not line:
                break

            x, y, kut = map(float, line.split())
            x1 = x / 20 * dimenzije
            y1 = dimenzije - (y / 20 * dimenzije)
            x2 = x1 + (0.5/2 * math.cos(kut)) / 20 * dimenzije
            y2 = y1 - (0.5/2 * math.sin(kut)) / 20 * dimenzije
            #x = (x1 + x2) / 2
            #y = (y1 + y2) / 2
            
            angle = kut / (2 * math.pi) * 360
            kutevi[i] = angle

            draw_line_at_angle(canvas, x2, y2, angle, linetag)
            
            i += 1
        upis_koordinata()
        
        evaluacija()#evaluacija mora bit prva jer ak je druga prebrise nacrtano

        

def cleanup():
    if old_image is not None and os.path.exists(old_image):
        os.remove(old_image)
    root.destroy()

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

    upis_koordinata()

# Create the main window
root = tk.Tk()
root.title("Mirror Placement")
root.protocol("WM_DELETE_WINDOW", cleanup)  # Bind the cleanup function to the close event

# Configure grid layout
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.rowconfigure(0, weight=1)  # First row
root.rowconfigure(1, weight=1)  # Second row

# Load an image using PIL
image = Image.open("baza.png")  # Replace with the path to your image
image = image.resize((dimenzije, dimenzije))  # Resize the image to fit the canvas
img = ImageTk.PhotoImage(image)

# Create a frame for the text fields
text_frame = ttk.Frame(root)
text_frame.grid(row=0, column=0, columnspan=2, pady=1, sticky="n")

# Create three text fields and place them side by side
text1 = ttk.Entry(text_frame, width=10)
text1.grid(row=0, column=0, padx=5)
text1.bind('<Return>', on_enter)  # Bind Enter key to all entries

text2 = ttk.Entry(text_frame, width=10)
text2.grid(row=0, column=1, padx=5)
text2.bind('<Return>', on_enter)  # Bind Enter key to all entries

text3 = ttk.Entry(text_frame, width=10)
text3.grid(row=0, column=2, padx=5)
text3.bind('<Return>', on_enter)  # Bind Enter key to all entries


# Create a canvas to display the image
canvas = tk.Canvas(root, width=dimenzije, height=dimenzije)
canvas.grid(row=1, column=0, padx=0, pady=0)
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
radio_frame.grid(row=1, column=1, padx=10, pady=10, sticky="n")

# Create 9 radio buttons and place them in the frame
radio_button = tk.Radiobutton(radio_frame, text="Lampa", variable=radio_var, value=0, command=on_select)
radio_button.pack(anchor="w")

for i in range(1, 9):
    radio_button = tk.Radiobutton(radio_frame, text=f"Zrcalo {i}", variable=radio_var, value=i, command=on_select)
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

loadaj_button = ttk.Button(radio_frame, text="Load File", command=load_file)
loadaj_button.pack(pady=10)

save_button = ttk.Button(radio_frame, text="Save File", command=save_file)
save_button.pack(pady=10)

# Run the main loop
root.mainloop()



