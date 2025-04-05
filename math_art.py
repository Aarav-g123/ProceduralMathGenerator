import numpy as np
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import ttk
import math
import random
from collections import deque

class FractalApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Fractal Explorer 2.0")
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
        
        tk.Label(self, text="Fractal Explorer 2.0", font=("Arial", 24)).pack(pady=20)
        
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
        fractals = [
            ("Barnsley Fern", lambda: self.app.show_fractal(IFSFractalFrame, 'barnsley')),
            ("Koch Curve", lambda: self.app.show_fractal(IFSFractalFrame, 'koch')),
            ("Dragon Curve", lambda: self.app.show_fractal(IFSFractalFrame, 'dragon')),
            ("Levy C Curve", lambda: self.app.show_fractal(IFSFractalFrame, 'levy')),
            ("Fractal Tree", lambda: self.app.show_fractal(IFSFractalFrame, 'tree'))
        ]
        for text, command in fractals:
            tk.Button(self, text=text, width=20, height=2, command=command).pack(pady=2)

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
        self.img = Image.new('RGB', (self.size, self.size), 'black')
        self.draw = ImageDraw.Draw(self.img)
        self.queue = deque()
        margin = 50
        start_points = [
            (self.size//2, margin),
            (margin, self.size - margin),
            (self.size - margin, self.size - margin)
        ]
        self.queue.append((start_points, 0))
        self.max_depth = 6
        self.animate_sierpinski()

    def animate_sierpinski(self):
        if not self.queue:
            self.tk_img = ImageTk.PhotoImage(image=self.img)
            self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)
            return
        
        next_queue = deque()
        while self.queue:
            points, depth = self.queue.popleft()
            if depth >= self.max_depth:
                self.draw.polygon(points, fill='white')
            else:
                mid_points = [
                    ((points[0][0] + points[1][0])/2, (points[0][1] + points[1][1])/2),
                    ((points[1][0] + points[2][0])/2, (points[1][1] + points[2][1])/2),
                    ((points[2][0] + points[0][0])/2, (points[2][1] + points[0][1])/2)
                ]
                sub_triangles = [
                    [points[0], mid_points[0], mid_points[2]],
                    [mid_points[0], points[1], mid_points[1]],
                    [mid_points[2], mid_points[1], points[2]]
                ]
                for triangle in sub_triangles:
                    next_queue.append((triangle, depth + 1))
        
        self.queue = next_queue
        self.tk_img = ImageTk.PhotoImage(image=self.img)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)
        self.after(100, self.animate_sierpinski)

class IFSFractalFrame(BaseFractal):
    def __init__(self, master, app, fractal_type):
        self.fractal_type = fractal_type
        super().__init__(master, app)

    def draw_fractal(self):
        transforms = []
        self.color = (34, 139, 34)  # Default green
        
        if self.fractal_type == 'barnsley':
            transforms = [
                (0.85, 0.04, -0.04, 0.85, 0, 1.6, 0.85),
                (0.2, -0.26, 0.23, 0.22, 0, 1.6, 0.07),
                (-0.15, 0.28, 0.26, 0.24, 0, 0.44, 0.07),
                (0, 0, 0, 0.16, 0, 0, 0.01)
            ]
        elif self.fractal_type == 'koch':
            transforms = [
                (0.3333, 0, 0, 0.3333, 0, 0, 0.25),
                (0.1667, -0.2887, 0.2887, 0.1667, 0.3333, 0, 0.25),
                (0.1667, 0.2887, -0.2887, 0.1667, 0.5, 0.2887, 0.25),
                (0.3333, 0, 0, 0.3333, 0.6667, 0, 0.25)
            ]
            self.color = (255, 255, 255)
        elif self.fractal_type == 'dragon':
            transforms = [
                (0.5, 0.5, -0.5, 0.5, 0, 0, 0.5),
                (-0.5, -0.5, 0.5, -0.5, 1, 0, 0.5)
            ]
            self.color = (0, 0, 255)
        elif self.fractal_type == 'levy':
            transforms = [
                (0.5, -0.5, 0.5, 0.5, 0, 0, 0.5),
                (0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5)
            ]
            self.color = (255, 0, 0)
        elif self.fractal_type == 'tree':
            transforms = [
                (0.195, -0.488, 0.344, 0.443, 0.4431, 0.2452, 0.4),
                (0.462, 0.414, -0.252, 0.361, 0.2511, 0.5692, 0.4),
                (-0.058, -0.07, 0.453, -0.111, 0.5976, 0.0969, 0.15),
                (-0.035, 0.07, -0.469, -0.022, 0.488, 0.5069, 0.05)
            ]
        
        # Set scaling and offsets
        if self.fractal_type == 'barnsley':
            scale_x = self.size / 11
            scale_y = -self.size / 11
            offset_x = self.size / 2
            offset_y = self.size
        elif self.fractal_type == 'tree':
            scale_x = self.size / 2.5
            scale_y = -self.size / 2.5
            offset_x = self.size / 2
            offset_y = self.size
        else:
            scale_x = self.size / 3
            scale_y = self.size / 3
            offset_x = self.size / 2
            offset_y = self.size / 2

        self.img = Image.new('RGB', (self.size, self.size))
        self.pixels = self.img.load()
        self.x, self.y = 0.0, 0.0
        self.total_points = 20000
        self.points_generated = 0
        self.batch_size = 100
        self.transforms = transforms
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.after(0, self.generate_batch)

    def generate_batch(self):
        if self.points_generated >= self.total_points:
            self.tk_img = ImageTk.PhotoImage(image=self.img)
            self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)
            return

        for _ in range(self.batch_size):
            if self.points_generated >= self.total_points:
                break
            r = random.random()
            cumulative = 0.0
            for a, b, c, d, e, f, prob in self.transforms:
                cumulative += prob
                if r < cumulative:
                    x_new = a * self.x + b * self.y + e
                    y_new = c * self.x + d * self.y + f
                    self.x, self.y = x_new, y_new
                    break

            px = int(self.x * self.scale_x + self.offset_x)
            py = int(self.y * self.scale_y + self.offset_y)
            if 0 <= px < self.size and 0 <= py < self.size:
                self.pixels[px, py] = self.color

            self.points_generated += 1

        self.tk_img = ImageTk.PhotoImage(image=self.img)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)
        self.after(10, self.generate_batch)

if __name__ == "__main__":
    root = tk.Tk()
    app = FractalApp(root)
    root.mainloop()