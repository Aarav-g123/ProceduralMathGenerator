import numpy as np
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import ttk
import math
import random

class FractalApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Fractal Explorer")
        self.master.geometry("800x800")
        self.current_frame = None
        self.show_main_menu()

    def show_main_menu(self):
        self.clear_current_frame()
        self.current_frame = MainMenu(self.master, self)
        self.current_frame.pack(fill=tk.BOTH, expand=True)

    def show_category(self, category_class):
        self.clear_current_frame()
        self.current_frame = category_class(self.master, self)
        self.current_frame.pack(fill=tk.BOTH, expand=True)

    def show_fractal(self, fractal_class, *args):
        self.clear_current_frame()
        self.current_frame = fractal_class(self.master, self, *args)
        self.current_frame.pack(fill=tk.BOTH, expand=True)

    def clear_current_frame(self):
        if self.current_frame:
            self.current_frame.destroy()

class MainMenu(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        
        tk.Label(self, text="Fractal Explorer", font=("Arial", 24)).pack(pady=20)
        
        categories = [
            ("Mandelbrot Family", MandelbrotCategory),
            ("Julia Sets", JuliaCategory),
            ("Newton Fractals", NewtonCategory),
            ("Sierpinski's Gasket", SierpinskiCategory),
            ("IFS Fractals", IFSCategory)
        ]
        
        for text, category_class in categories:
            btn = tk.Button(self, text=text, width=20, height=2,
                          command=lambda c=category_class: app.show_category(c))
            btn.pack(pady=2)
        
        tk.Button(self, text="Quit", width=20, height=2,
                command=master.destroy, bg="#ff9999").pack(pady=10)

class BaseCategory(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.create_ui()

    def create_ui(self):
        tk.Button(self, text="← Back", command=self.app.show_main_menu).pack(anchor="nw")
        self.create_fractal_buttons()

    def create_fractal_buttons(self):
        raise NotImplementedError("Subclasses must implement this method")

class MandelbrotCategory(BaseCategory):
    def create_fractal_buttons(self):
        fractals = [
            ("Classic Mandelbrot", MandelbrotFrame),
            ("Burning Ship", BurningShipFrame)
        ]
        for text, fractal_class in fractals:
            tk.Button(self, text=text, width=20, height=2,
                    command=lambda c=fractal_class: self.app.show_fractal(c)).pack(pady=2)

class JuliaCategory(BaseCategory):
    def create_fractal_buttons(self):
        julia_sets = [
            ("Julia Set (c=-0.4+0.6i)", complex(-0.4, 0.6)),
            ("Julia Set (c=0.285+0.01i)", complex(0.285, 0.01))
        ]
        for text, constant in julia_sets:
            tk.Button(self, text=text, width=20, height=2,
                    command=lambda c=constant: self.app.show_fractal(JuliaFrame, c)).pack(pady=2)

class NewtonCategory(BaseCategory):
    def create_fractal_buttons(self):
        tk.Button(self, text="Newton Fractal (z³-1)", width=20, height=2,
                command=lambda: self.app.show_fractal(NewtonFrame)).pack(pady=2)

class SierpinskiCategory(BaseCategory):
    def create_fractal_buttons(self):
        tk.Button(self, text="Sierpinski Triangle", width=20, height=2,
                command=lambda: self.app.show_fractal(SierpinskiFrame)).pack(pady=2)

class IFSCategory(BaseCategory):
    def create_fractal_buttons(self):
        tk.Button(self, text="Barnsley Fern", width=20, height=2,
                command=lambda: self.app.show_fractal(BarnsleyFernFrame)).pack(pady=2)

class BaseFractal(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.size = 600
        self.iterations = 50
        self.setup_ui()

    def setup_ui(self):
        tk.Button(self, text="← Back", command=self.app.show_main_menu).pack(anchor="nw")
        self.canvas = tk.Canvas(self, width=self.size, height=self.size)
        self.canvas.pack()
        self.create_controls()
        self.draw_fractal()

    def create_controls(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X)
        ttk.Button(control_frame, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=5)
        ttk.Label(control_frame, text="Iterations:").pack(side=tk.LEFT)
        self.iter_slider = ttk.Scale(control_frame, from_=10, to=200, 
                                   command=lambda v: self.set_iterations(int(float(v))))
        self.iter_slider.set(self.iterations)
        self.iter_slider.pack(side=tk.LEFT, padx=5)

    def set_iterations(self, iterations):
        self.iterations = iterations
        self.draw_fractal()

    def reset(self):
        self.draw_fractal()

    def draw_fractal(self):
        raise NotImplementedError("Subclasses must implement this method")

class MandelbrotFrame(BaseFractal):
    def draw_fractal(self):
        x = np.linspace(-2, 1, self.size)
        y = np.linspace(-1.5, 1.5, self.size)
        c = x[:, None] + 1j * y[None, :]
        z = np.zeros_like(c)
        div_time = np.full(c.shape, self.iterations, dtype=np.float32)

        for i in range(self.iterations):
            mask = (div_time == self.iterations)
            z[mask] = z[mask] ** 2 + c[mask]
            div_time[(np.abs(z) > 2) & mask] = i

        hue = (div_time.T * 255 / self.iterations).astype(np.uint8)
        img = Image.fromarray(hue).convert('RGB')
        self.tk_img = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

class BurningShipFrame(BaseFractal):
    def draw_fractal(self):
        x = np.linspace(-2, 1, self.size)
        y = np.linspace(-1.5, 1.5, self.size)
        c = x[:, None] + 1j * y[None, :]
        z = np.zeros_like(c)
        div_time = np.full(c.shape, self.iterations, dtype=np.float32)

        for i in range(self.iterations):
            mask = (div_time == self.iterations)
            z[mask] = (np.abs(z[mask].real) + 1j * np.abs(z[mask].imag)) ** 2 + c[mask]
            div_time[(np.abs(z) > 2) & mask] = i

        hue = (div_time.T * 255 / self.iterations).astype(np.uint8)
        img = Image.fromarray(hue).convert('RGB')
        self.tk_img = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

class JuliaFrame(BaseFractal):
    def __init__(self, master, app, constant):
        self.constant = constant
        super().__init__(master, app)

    def draw_fractal(self):
        x = np.linspace(-2, 2, self.size)
        y = np.linspace(-2, 2, self.size)
        z = x[:, None] + 1j * y[None, :]
        div_time = np.full(z.shape, self.iterations, dtype=np.float32)

        for i in range(self.iterations):
            mask = (div_time == self.iterations)
            z[mask] = z[mask] ** 2 + self.constant
            div_time[(np.abs(z) > 2) & mask] = i

        hue = (div_time.T * 255 / self.iterations).astype(np.uint8)
        img = Image.fromarray(hue).convert('RGB')
        self.tk_img = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

class NewtonFrame(BaseFractal):
    def draw_fractal(self):
        x = np.linspace(-2, 2, self.size)
        y = np.linspace(-2, 2, self.size)
        X, Y = np.meshgrid(x, y)
        Z = X + Y * 1j

        for _ in range(self.iterations):
            mask = np.abs(Z) > 1e-6
            Z[mask] -= (Z[mask] ** 3 - 1) / (3 * Z[mask] ** 2)

        roots = [1, -0.5+0.866j, -0.5-0.866j]
        colors = np.array([[255,0,0], [0,255,0], [0,0,255], [100,100,100]])
        distances = np.abs(Z[:,:,None] - np.array(roots)[None,None,:])
        closest = np.argmin(distances, axis=2)
        img_array = colors[closest]

        self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(img_array.astype(np.uint8)))
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

class SierpinskiFrame(BaseFractal):
    def draw_fractal(self):
        img = Image.new('RGB', (self.size, self.size), 'black')
        draw = ImageDraw.Draw(img)
        
        def draw_triangle(points, depth):
            if depth <= 0:
                draw.polygon(points, fill='white')
                return
            
            mid_points = [
                ((points[0][0] + points[1][0])/2, (points[0][1] + points[1][1])/2),
                ((points[1][0] + points[2][0])/2, (points[1][1] + points[2][1])/2),
                ((points[2][0] + points[0][0])/2, (points[2][1] + points[0][1])/2)
            ]
            
            draw_triangle([points[0], mid_points[0], mid_points[2]], depth-1)
            draw_triangle([mid_points[0], points[1], mid_points[1]], depth-1)
            draw_triangle([mid_points[2], mid_points[1], points[2]], depth-1)

        margin = 50
        start_points = [
            (self.size//2, margin),
            (margin, self.size - margin),
            (self.size - margin, self.size - margin)
        ]
        
        draw_triangle(start_points, 6)
        self.tk_img = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

class BarnsleyFernFrame(BaseFractal):
    def draw_fractal(self):
        transforms = [
            (0.85, 0.04, -0.04, 0.85, 0, 1.6, 0.85),
            (0.2, -0.26, 0.23, 0.22, 0, 1.6, 0.07),
            (-0.15, 0.28, 0.26, 0.24, 0, 0.44, 0.07),
            (0, 0, 0, 0.16, 0, 0, 0.01)
        ]
        
        x, y = 0, 0
        points = np.zeros((20000, 2))
        for i in range(len(points)):
            r = random.random()
            for a, b, c, d, e, f, prob in transforms:
                if r < prob:
                    x, y = a*x + b*y + e, c*x + d*y + f
                    break
            points[i] = [x, y]
        
        points = (points * [self.size/11, -self.size/11] + [self.size/2, self.size]).astype(int)
        valid = (points[:,0] >= 0) & (points[:,0] < self.size) & (points[:,1] >= 0) & (points[:,1] < self.size)
        
        img = Image.new('RGB', (self.size, self.size))
        pixels = img.load()
        for x, y in points[valid]:
            pixels[x, y] = (34, 139, 34)
        
        self.tk_img = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

if __name__ == "__main__":
    root = tk.Tk()
    app = FractalApp(root)
    root.mainloop()