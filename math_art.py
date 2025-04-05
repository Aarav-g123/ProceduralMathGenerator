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
        self.master.title("Fractal Explorer 3.0")
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
        
        tk.Label(self, text="Fractal Explorer 3.0", font=("Arial", 24)).pack(pady=20)
        
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
            ("Julia Set (c=0.285+0.01i)", complex(0.285, 0.01)),
            ("Julia Set (c=-0.8+0.156i)", complex(-0.8, 0.156))
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
            ("Koch Snowflake", lambda: self.app.show_fractal(IFSFractalFrame, 'koch')),
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
        self.animating = False
        self.animation_delay = 50
        self.setup_ui()

    def setup_ui(self):
        tk.Button(self, text="← Back", command=self.app.show_main_menu).pack(anchor="nw")
        self.canvas = tk.Canvas(self, width=self.size, height=self.size, bg='black')
        self.canvas.pack()
        self.create_controls()
        self.start_animation()

    def create_controls(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X)
        
        self.start_stop = ttk.Button(control_frame, text="Stop", command=self.toggle_animation)
        self.start_stop.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="Speed:").pack(side=tk.LEFT)
        self.speed_slider = ttk.Scale(control_frame, from_=1, to=100, 
                                    command=lambda v: self.set_speed(101-float(v)))
        self.speed_slider.set(100 - self.animation_delay)
        self.speed_slider.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="Iterations:").pack(side=tk.LEFT)
        self.iter_slider = ttk.Scale(control_frame, from_=10, to=200, 
                                   command=lambda v: self.set_iterations(int(float(v))))
        self.iter_slider.set(self.iterations)
        self.iter_slider.pack(side=tk.LEFT, padx=5)

    def toggle_animation(self):
        self.animating = not self.animating
        self.start_stop.config(text="Start" if not self.animating else "Stop")
        if self.animating:
            self.start_animation()

    def set_speed(self, delay):
        self.animation_delay = int(delay)

    def set_iterations(self, iterations):
        self.iterations = iterations
        self.reset()

    def reset(self):
        self.animating = False
        self.start_stop.config(text="Start")
        self.canvas.delete("all")
        if hasattr(self, 'anim_id'):
            self.after_cancel(self.anim_id)
        self.start_animation()

    def start_animation(self):
        raise NotImplementedError("Subclasses must implement this method")

class MandelbrotFrame(BaseFractal):
    def start_animation(self):
        self.x = np.linspace(-2, 1, self.size)
        self.y = np.linspace(-1.5, 1.5, self.size)
        self.c = self.x[:, None] + 1j * self.y[None, :]
        self.z = np.zeros_like(self.c)
        self.div_time = np.full(self.c.shape, self.iterations, dtype=np.float32)
        self.current_iter = 0
        self.animating = True
        self.animate()

    def animate(self):
        if not self.animating or self.current_iter >= self.iterations:
            return

        mask = (self.div_time == self.iterations)
        self.z[mask] = self.z[mask] ** 2 + self.c[mask]
        self.div_time[(np.abs(self.z) > 2) & mask] = self.current_iter
        
        hue = (self.div_time.T * 255 / self.iterations).astype(np.uint8)
        img = Image.fromarray(hue).convert('RGB')
        self.tk_img = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)
        
        self.current_iter += 1
        self.anim_id = self.after(self.animation_delay, self.animate)

class BurningShipFrame(BaseFractal):
    def start_animation(self):
        self.x = np.linspace(-2, 1, self.size)
        self.y = np.linspace(-1.5, 1.5, self.size)
        self.c = self.x[:, None] + 1j * self.y[None, :]
        self.z = np.zeros_like(self.c)
        self.div_time = np.full(self.c.shape, self.iterations, dtype=np.float32)
        self.current_iter = 0
        self.animating = True
        self.animate()

    def animate(self):
        if not self.animating or self.current_iter >= self.iterations:
            return

        mask = (self.div_time == self.iterations)
        self.z[mask] = (np.abs(self.z[mask].real) + 1j * np.abs(self.z[mask].imag)) ** 2 + self.c[mask]
        self.div_time[(np.abs(self.z) > 2) & mask] = self.current_iter
        
        hue = (self.div_time.T * 255 / self.iterations).astype(np.uint8)
        img = Image.fromarray(hue).convert('RGB')
        self.tk_img = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)
        
        self.current_iter += 1
        self.anim_id = self.after(self.animation_delay, self.animate)

class JuliaFrame(BaseFractal):
    def __init__(self, master, app, constant):
        self.constant = constant
        super().__init__(master, app)

    def start_animation(self):
        self.x = np.linspace(-2, 2, self.size)
        self.y = np.linspace(-2, 2, self.size)
        self.z = self.x[:, None] + 1j * self.y[None, :]
        self.div_time = np.full(self.z.shape, self.iterations, dtype=np.float32)
        self.current_iter = 0
        self.animating = True
        self.animate()

    def animate(self):
        if not self.animating or self.current_iter >= self.iterations:
            return

        mask = (self.div_time == self.iterations)
        self.z[mask] = self.z[mask] ** 2 + self.constant
        self.div_time[(np.abs(self.z) > 2) & mask] = self.current_iter
        
        hue = (self.div_time.T * 255 / self.iterations).astype(np.uint8)
        img = Image.fromarray(hue).convert('RGB')
        self.tk_img = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)
        
        self.current_iter += 1
        self.anim_id = self.after(self.animation_delay, self.animate)

class NewtonFrame(BaseFractal):
    def start_animation(self):
        self.x = np.linspace(-2, 2, self.size)
        self.y = np.linspace(-2, 2, self.size)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.Z = self.X + self.Y * 1j
        self.current_iter = 0
        self.animating = True
        self.animate()

    def animate(self):
        if not self.animating or self.current_iter >= self.iterations:
            return

        mask = np.abs(self.Z) > 1e-6
        self.Z[mask] -= (self.Z[mask] ** 3 - 1) / (3 * self.Z[mask] ** 2)
        
        roots = [1, -0.5+0.866j, -0.5-0.866j]
        colors = np.array([[255,0,0], [0,255,0], [0,0,255], [100,100,100]])
        distances = np.abs(self.Z[:,:,None] - np.array(roots)[None,None,:])
        closest = np.argmin(distances, axis=2)
        img_array = colors[closest]

        self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(img_array.astype(np.uint8)))
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)
        
        self.current_iter += 1
        self.anim_id = self.after(self.animation_delay, self.animate)

class SierpinskiFrame(BaseFractal):
    def start_animation(self):
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
        self.animating = True
        self.animate()

    def animate(self):
        if not self.animating or not self.queue:
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
        self.anim_id = self.after(self.animation_delay, self.animate)

class IFSFractalFrame(BaseFractal):
    def __init__(self, master, app, fractal_type):
        self.fractal_type = fractal_type
        super().__init__(master, app)

    def start_animation(self):
        self.configure_transforms()
        self.img = Image.new('RGB', (self.size, self.size), 'black')
        self.pixels = self.img.load()
        self.x, self.y = 0.0, 0.0
        self.points_generated = 0
        self.animating = True
        self.animate()

    def configure_transforms(self):
        if self.fractal_type == 'barnsley':
            self.transforms = [
                (0.85, 0.04, -0.04, 0.85, 0, 1.6, 0.85),
                (0.2, -0.26, 0.23, 0.22, 0, 1.6, 0.07),
                (-0.15, 0.28, 0.26, 0.24, 0, 0.44, 0.07),
                (0, 0, 0, 0.16, 0, 0, 0.01)
            ]
            self.color = (34, 139, 34)
            self.scale_x = self.size / 11
            self.scale_y = -self.size / 11
            self.offset_x = self.size / 2
            self.offset_y = self.size
        elif self.fractal_type == 'koch':
            angle = math.pi / 3  # 60 degrees
            self.transforms = [
                (1/3, 0, 0, 1/3, 0, 0, 0.25),
                (1/3*math.cos(angle), -1/3*math.sin(angle),
                 1/3*math.sin(angle), 1/3*math.cos(angle),
                 1/3, 0, 0.25),
                (1/3*math.cos(-angle), -1/3*math.sin(-angle),
                 1/3*math.sin(-angle), 1/3*math.cos(-angle),
                 1/2, 1/3*math.sin(angle), 0.25),
                (1/3, 0, 0, 1/3, 2/3, 0, 0.25)
            ]
            self.color = (255, 255, 255)
            self.scale_x = self.size
            self.scale_y = self.size
            self.offset_x = 0
            self.offset_y = self.size/2
        elif self.fractal_type == 'dragon':
            self.transforms = [
                (0.5, 0.5, -0.5, 0.5, 0, 0, 0.5),
                (-0.5, -0.5, 0.5, -0.5, 1, 0, 0.5)
            ]
            self.color = (0, 0, 255)
            self.scale_x = self.size / 3
            self.scale_y = self.size / 3
            self.offset_x = self.size / 2
            self.offset_y = self.size / 2
        elif self.fractal_type == 'levy':
            self.transforms = [
                (0.5, -0.5, 0.5, 0.5, 0, 0, 0.5),
                (0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5)
            ]
            self.color = (255, 0, 0)
            self.scale_x = self.size / 2
            self.scale_y = self.size / 2
            self.offset_x = self.size / 4
            self.offset_y = self.size / 4
        elif self.fractal_type == 'tree':
            self.transforms = [
                (0.195, -0.488, 0.344, 0.443, 0.4431, 0.2452, 0.4),
                (0.462, 0.414, -0.252, 0.361, 0.2511, 0.5692, 0.4),
                (-0.058, -0.07, 0.453, -0.111, 0.5976, 0.0969, 0.15),
                (-0.035, 0.07, -0.469, -0.022, 0.488, 0.5069, 0.05)
            ]
            self.color = (0, 200, 0)
            self.scale_x = self.size / 2.5
            self.scale_y = -self.size / 2.5
            self.offset_x = self.size / 2
            self.offset_y = self.size

    def animate(self):
        if not self.animating or self.points_generated >= 20000:
            return

        for _ in range(100):
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
        self.anim_id = self.after(self.animation_delay, self.animate)

if __name__ == "__main__":
    root = tk.Tk()
    app = FractalApp(root)
    root.mainloop()