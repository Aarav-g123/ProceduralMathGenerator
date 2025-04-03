import numpy as np
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import ttk
import math

class FractalApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Fractal Explorer")
        self.master.geometry("800x800")
        self.current_frame = None
        self.show_start_screen()
        
    def show_start_screen(self):
        self.clear_current_frame()
        self.current_frame = StartScreen(self.master, self)
        self.current_frame.pack(fill=tk.BOTH, expand=True)

    def show_fractal(self, fractal_type):
        self.clear_current_frame()
        if fractal_type == "mandelbrot":
            self.current_frame = MandelbrotSubMenu(self.master, self)
        elif fractal_type == "sierpinski":
            self.current_frame = SierpinskiFrame(self.master, self)
        self.current_frame.pack(fill=tk.BOTH, expand=True)

    def show_fractal_type(self, fractal_class):
        self.clear_current_frame()
        self.current_frame = fractal_class(self.master, self)
        self.current_frame.pack(fill=tk.BOTH, expand=True)

    def clear_current_frame(self):
        if self.current_frame:
            self.current_frame.destroy()

class StartScreen(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        
        title = tk.Label(self, text="Fractal Explorer", font=("Arial", 24))
        title.pack(pady=20)
        
        buttons = [
            ("Mandelbrot Family", lambda: app.show_fractal("mandelbrot")),
            ("Sierpinski Triangle", lambda: app.show_fractal("sierpinski")),
            ("Coming Soon...", None)
        ]
        
        for text, command in buttons:
            btn = tk.Button(self, text=text, width=20, height=2,
                          command=command if command else lambda: None)
            btn.pack(pady=10)

class MandelbrotSubMenu(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        
        tk.Button(self, text="← Back", command=app.show_start_screen).pack(anchor="nw")
        title = tk.Label(self, text="Mandelbrot Family", font=("Arial", 20))
        title.pack(pady=20)
        
        types = [
            ("Standard Mandelbrot", MandelbrotFrame),
            ("Julia Set", JuliaFrame),
            ("Burning Ship", BurningShipFrame)
        ]
        
        for text, fractal_class in types:
            btn = tk.Button(self, text=text, width=20, height=2,
                           command=lambda c=fractal_class: app.show_fractal_type(c))
            btn.pack(pady=10)

class BaseFractalFrame(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.size = 600
        self.iterations = 100
        self.xmin, self.xmax = -2.0, 1.0
        self.ymin, self.ymax = -1.5, 1.5
        self.dragging = False
        self.start_x, self.start_y = 0, 0
        self.selection_rect = None
        
        self.create_widgets()
    
    def create_widgets(self):

        nav_frame = tk.Frame(self)
        nav_frame.pack(fill=tk.X)
        tk.Button(nav_frame, text="← Back", command=self.app.show_start_screen).pack(side=tk.LEFT)
        

        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X)
        
        ttk.Button(control_frame, text="Reset", command=self.reset_view).pack(side=tk.LEFT, padx=5)
        ttk.Label(control_frame, text="Iterations:").pack(side=tk.LEFT)
        self.iter_slider = ttk.Scale(control_frame, from_=50, to=500, 
                                   command=lambda v: self.set_iterations(int(float(v))))
        self.iter_slider.set(self.iterations)
        self.iter_slider.pack(side=tk.LEFT, padx=5)
        

        self.canvas = tk.Canvas(self, width=self.size, height=self.size)
        self.canvas.pack()
        
 
        self.canvas.bind("<ButtonPress-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.update_selection)
        self.canvas.bind("<ButtonRelease-1>", self.end_drag)
        self.canvas.bind("<MouseWheel>", self.zoom)
    
    def set_iterations(self, iterations):
        self.iterations = iterations
        self.draw_fractal()
    
    def reset_view(self):
        self.xmin, self.xmax = -2.0, 1.0
        self.ymin, self.ymax = -1.5, 1.5
        self.draw_fractal()
    
    def start_drag(self, event):
        self.dragging = True
        self.start_x, self.start_y = event.x, event.y
        self.selection_rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline='white')
    
    def update_selection(self, event):
        if self.dragging and self.selection_rect:
            self.canvas.coords(self.selection_rect, 
                             self.start_x, self.start_y, 
                             event.x, event.y)
    
    def end_drag(self, event):
        if not self.dragging or not self.selection_rect:
            return
        
        self.dragging = False
        x1, y1 = self.start_x, self.start_y
        x2, y2 = event.x, event.y
        

        new_xmin = self.xmin + (min(x1, x2)/self.size) * (self.xmax - self.xmin)
        new_xmax = self.xmin + (max(x1, x2)/self.size) * (self.xmax - self.xmin)
        new_ymin = self.ymin + (min(y1, y2)/self.size) * (self.ymax - self.ymin)
        new_ymax = self.ymin + (max(y1, y2)/self.size) * (self.ymax - self.ymin)
        
        self.xmin, self.xmax = new_xmin, new_xmax
        self.ymin, self.ymax = new_ymin, new_ymax
        self.canvas.delete(self.selection_rect)
        self.selection_rect = None
        self.draw_fractal()
    
    def zoom(self, event):
        factor = 0.5 if event.delta > 0 else 2.0
        mouse_x = event.x
        mouse_y = event.y
        

        current_width = self.xmax - self.xmin
        current_height = self.ymax - self.ymin
        target_x = self.xmin + (mouse_x/self.size) * current_width
        target_y = self.ymin + (mouse_y/self.size) * current_height
        

        new_width = current_width * factor
        new_height = current_height * factor
        
        self.xmin = target_x - new_width/2
        self.xmax = target_x + new_width/2
        self.ymin = target_y - new_height/2
        self.ymax = target_y + new_height/2
        
        self.draw_fractal()
    
    def draw_fractal(self):
        raise NotImplementedError("Subclasses must implement this method")

class MandelbrotFrame(BaseFractalFrame):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.draw_fractal()
    
    def draw_fractal(self):
        x = np.linspace(self.xmin, self.xmax, self.size)
        y = np.linspace(self.ymin, self.ymax, self.size)
        c = x[:, np.newaxis] + 1j * y[np.newaxis, :]
        
        z = np.zeros_like(c, dtype=np.complex128)
        div_time = np.zeros(c.shape, dtype=int)
        
        for i in range(self.iterations):
            mask = (div_time == 0)
            z[mask] = z[mask]**2 + c[mask]
            diverged = np.abs(z) >= 2
            div_time[diverged & (div_time == 0)] = i
        
        hue = (div_time.T * 255 // self.iterations).astype(np.uint8)
        img = Image.fromarray(hue, mode='L').convert('RGB')
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

class JuliaFrame(BaseFractalFrame):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.c = complex(-0.4, 0.6)
        self.draw_fractal()
    
    def draw_fractal(self):
        x = np.linspace(self.xmin, self.xmax, self.size)
        y = np.linspace(self.ymin, self.ymax, self.size)
        z = x[:, np.newaxis] + 1j * y[np.newaxis, :]
        
        div_time = np.zeros(z.shape, dtype=int)
        for i in range(self.iterations):
            mask = (div_time == 0)
            z[mask] = z[mask]**2 + self.c
            diverged = np.abs(z) >= 2
            div_time[diverged & (div_time == 0)] = i
        
        hue = (div_time.T * 255 // self.iterations).astype(np.uint8)
        img = Image.fromarray(hue, mode='L').convert('RGB')
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

class BurningShipFrame(BaseFractalFrame):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.draw_fractal()
    
    def draw_fractal(self):
        x = np.linspace(self.xmin, self.xmax, self.size)
        y = np.linspace(self.ymin, self.ymax, self.size)
        c = x[:, np.newaxis] + 1j * y[np.newaxis, :]
        
        z = np.zeros_like(c, dtype=np.complex128)
        div_time = np.zeros(c.shape, dtype=int)
        
        for i in range(self.iterations):
            mask = (div_time == 0)
            z[mask] = (np.abs(z[mask].real) + 1j * np.abs(z[mask].imag))**2 + c[mask]
            diverged = np.abs(z) >= 2
            div_time[diverged & (div_time == 0)] = i
        
        hue = (div_time.T * 255 // self.iterations).astype(np.uint8)
        img = Image.fromarray(hue, mode='L').convert('RGB')
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

class SierpinskiFrame(BaseFractalFrame):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.iter_slider.pack_forget()
        self.draw_fractal()
    
    def draw_fractal(self):
        img = Image.new('RGB', (self.size, self.size), color='black')
        draw = ImageDraw.Draw(img)
        
        def draw_triangle(points, depth=7):
            if depth == 0:
                draw.polygon(points, fill='white')
                return
            
            mid1 = ((points[0][0] + points[1][0])/2, (points[0][1] + points[1][1])/2)
            mid2 = ((points[1][0] + points[2][0])/2, (points[1][1] + points[2][1])/2)
            mid3 = ((points[2][0] + points[0][0])/2, (points[2][1] + points[0][1])/2)
            
            draw_triangle([points[0], mid1, mid3], depth-1)
            draw_triangle([mid1, points[1], mid2], depth-1)
            draw_triangle([mid3, mid2, points[2]], depth-1)
        
        margin = 20
        start_points = [
            (self.size//2, margin),
            (margin, self.size - margin),
            (self.size - margin, self.size - margin)
        ]
        
        draw_triangle(start_points)
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

if __name__ == "__main__":
    root = tk.Tk()
    app = FractalApp(root)
    root.mainloop()