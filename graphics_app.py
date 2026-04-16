import tkinter as tk

def main():
    root = tk.Tk()
    root.title("Basic Graphical Primitives")

    canvas = tk.Canvas(root, width=400, height=400, bg="white")
    canvas.pack()

    # Draw a line
    canvas.create_line(50, 50, 200, 50, fill="blue", width=2)

    # Draw a rectangle
    canvas.create_rectangle(50, 100, 200, 200, outline="red", fill="yellow", width=2)

    # Draw an oval (circle)
    canvas.create_oval(250, 100, 350, 200, outline="green", fill="orange", width=2)

    # Draw text
    canvas.create_text(200, 300, text="Basic Graphical Primitives", fill="black", font=("Helvetica", 16, "bold"))

    root.mainloop()

if __name__ == "__main__":
    main()
