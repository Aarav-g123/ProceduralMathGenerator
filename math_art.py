import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import math
import random
from collections import deque
from functools import partial

#Mandelbulb :)


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
            ("IFS Fractals", IFSCategory),
            ("3D Fractals", ThreeDFractalsCategory)
        ]
        
        for text, category_class in categories:
            btn = tk.Button(self, text=text, width=20, height=2,
                          command=lambda c=category_class: self.app.show_category(c))
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
        tk.Button(self, text="Sierpinski Triangle (Recursive)", 
                 command=lambda: self.app.show_fractal(SierpinskiFrame)).pack(pady=2)
        tk.Button(self, text="Sierpinski Triangle (Chaos Game)", 
                 command=lambda: self.app.show_fractal(ChaosGameSierpinskiFrame)).pack(pady=2)
class ThreeDFractalsCategory(BaseCategory):
    def create_fractal_buttons(self):
        tk.Button(self, text="Mandelbulb", width=20, height=2,
                command=lambda: self.app.show_fractal(MandelbulbFrame)).pack(pady=2)

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
    def __init__(self, master, app, *args):
        super().__init__(master)
        self.xmin, self.xmax = -2.0, 2.0
        self.ymin, self.ymax = -2.0, 2.0
        self.app = app
        self.size = 600
        self.iterations = 100
        self.animating = False
        self.animation_delay = 50
        self.zoom_stack = []
        self.zoom_enabled = True
        self.setup_ui()
        self.setup_zoom()

    def setup_zoom(self):
        if self.zoom_enabled:
            self.canvas.bind("<ButtonPress-1>", self.start_zoom)
            self.canvas.bind("<B1-Motion>", self.update_zoom_box)
            self.canvas.bind("<ButtonRelease-1>", self.finish_zoom)
            self.master.bind("<Control-z>", self.zoom_out)
            self.master.bind("<Control-Z>", self.zoom_out)
        else:
            self.canvas.unbind("<ButtonPress-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            self.master.unbind("<Control-z>")
            self.master.unbind("<Control-Z>")
    def destroy(self):
        if hasattr(self, 'anim_id'):
            self.after_cancel(self.anim_id)
        super().destroy()
    def setup_ui(self):
        tk.Button(self, text="← Back", command=self.app.show_main_menu).pack(anchor="nw")
        self.canvas = tk.Canvas(self, width=self.size, height=self.size, bg='black')
        self.canvas.pack()
        self.create_controls()
        self.initialize_fractal()
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
        
        ttk.Button(control_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT, padx=5)

    def start_zoom(self, event):
        self.zoom_start = (event.x, event.y)
        self.zoom_rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline='white'
        )

    def update_zoom_box(self, event):
        self.canvas.coords(self.zoom_rect, self.zoom_start[0], self.zoom_start[1], event.x, event.y)

    def finish_zoom(self, event):
        x0, y0 = self.zoom_start
        x1, y1 = event.x, event.y
        if abs(x1 - x0) > 10 and abs(y1 - y0) > 10:
            self.zoom_stack.append((self.xmin, self.xmax, self.ymin, self.ymax))
            self.update_viewport(x0, y0, x1, y1)
            self.reset()
        self.canvas.delete(self.zoom_rect)

    def update_viewport(self, x0, y0, x1, y1):
        new_xmin = min(self.px_to_complex(x0, y0).real, self.px_to_complex(x1, y1).real)
        new_xmax = max(self.px_to_complex(x0, y0).real, self.px_to_complex(x1, y1).real)
        new_ymin = min(self.px_to_complex(x0, y0).imag, self.px_to_complex(x1, y1).imag)
        new_ymax = max(self.px_to_complex(x0, y0).imag, self.px_to_complex(x1, y1).imag)
        
        self.xmin, self.xmax = new_xmin, new_xmax
        self.ymin, self.ymax = new_ymin, new_ymax

    def zoom_out(self, event=None):
        if self.zoom_stack:
            self.xmin, self.xmax, self.ymin, self.ymax = self.zoom_stack.pop()
            self.reset()

    def px_to_complex(self, x, y):
        return complex(
            self.xmin + (x / self.size) * (self.xmax - self.xmin),
            self.ymin + (y / self.size) * (self.ymax - self.ymin)
        )

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
        self.initialize_fractal()
        self.start_animation()

    def initialize_fractal(self):
        raise NotImplementedError("Subclasses must implement this method")

    def start_animation(self):
        raise NotImplementedError("Subclasses must implement this method")

class ChaosGameSierpinskiFrame(BaseFractal):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.zoom_enabled = False
        self.xmin, self.xmax = 0.0, 1.0  
        self.ymin, self.ymax = 0.0, 1.0

    def initialize_fractal(self):

        self.vertices = [
            (0.5, 0.1), 
            (0.1, 0.9),  
            (0.9, 0.9)   
        ]
        self.current_point = (random.random(), random.random())
        

        self.img = Image.new('RGB', (self.size, self.size), 'black')
        self.draw = ImageDraw.Draw(self.img)
        

        for v in self.vertices:
            screen_x = int(v[0] * self.size)
            screen_y = int(v[1] * self.size)
            self.draw.rectangle([screen_x-2, screen_y-2, screen_x+2, screen_y+2], fill='red')
        

        start_x = int(self.current_point[0] * self.size)
        start_y = int(self.current_point[1] * self.size)
        self.draw.rectangle([start_x-2, start_y-2, start_x+2, start_y+2], fill='blue')

    def start_animation(self):
        self.animating = True
        self.animate()

    def animate(self):
        if not self.animating:
            return
        

        for _ in range(100):
            target = random.choice(self.vertices)
            new_x = (self.current_point[0] + target[0]) / 2
            new_y = (self.current_point[1] + target[1]) / 2
            self.current_point = (new_x, new_y)
            

            px = int(new_x * self.size)
            py = int(new_y * self.size)
            if 0 <= px < self.size and 0 <= py < self.size:
                self.draw.point((px, py), fill='white')

        self.tk_img = ImageTk.PhotoImage(image=self.img)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)
        self.anim_id = self.after(self.animation_delay, self.animate)
class MandelbulbFrame(BaseFractal):
    def __init__(self, master, app):
        # Initialize critical attributes first
        self.power = 8
        self.size = 300
        self.iterations = 30
        
        # Call parent constructor
        super().__init__(master, app)
        
        # Initialize remaining attributes
        self.zoom_enabled = False
        self.cam_pos = np.array([0, 0, -2.5])
        self.light_pos = np.array([2, 1, -1])
        self.rotation = 0.0
        self.rendering = False
        self.active = True
        
        # Clear existing UI from parent and rebuild
        self.rebuild_ui()

    def rebuild_ui(self):
        """Destroy parent UI elements and create new ones"""
        for widget in self.winfo_children():
            widget.destroy()
        
        self.setup_ui()
        self.create_3d_controls()

    def setup_ui(self):
        """Custom UI setup for Mandelbulb"""
        tk.Button(self, text="← Back", command=self.app.show_main_menu).pack(anchor="nw")
        self.canvas = tk.Canvas(self, width=self.size, height=self.size, bg='black')
        self.canvas.pack()
        self.initialize_fractal()

    def create_3d_controls(self):
        """Create Mandelbulb-specific controls"""
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X)
        
        self.start_stop = ttk.Button(control_frame, text="Start", command=self.toggle_animation)
        self.start_stop.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Reset", command=self.safe_reset).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="Power:").pack(side=tk.LEFT)
        self.power_slider = ttk.Scale(control_frame, from_=2, to=10, 
                                    command=lambda v: self.set_power(int(float(v))))
        self.power_slider.set(self.power)
        self.power_slider.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Rotate Left", 
                 command=lambda: self.update_rotation(0.1)).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Rotate Right", 
                 command=lambda: self.update_rotation(-0.1)).pack(side=tk.LEFT, padx=5)

    def set_power(self, power):
        self.power = power
        self.safe_reset()
        
    def update_rotation(self, angle):
        self.rotation += angle
        self.safe_reset()
        
    def initialize_fractal(self):
        self.img = Image.new('RGB', (self.size, self.size), 'black')
        self.pixels = self.img.load()
        
    def toggle_animation(self):
        if not self.rendering:
            self.start_animation()
        else:
            self.rendering = False
            self.start_stop.config(text="Start")

    def start_animation(self):
        if not self.active or self.rendering:
            return
            
        self.rendering = True
        self.start_stop.config(text="Stop")
        self.render_fractal()
        
    def render_fractal(self):
        if not self.active:
            return

        try:
            rot_mat = np.array([
                [math.cos(self.rotation), 0, -math.sin(self.rotation)],
                [0, 1, 0],
                [math.sin(self.rotation), 0, math.cos(self.rotation)]
            ])

            for y in range(self.size):
                if not self.active or not self.rendering:
                    break
                    
                for x in range(self.size):
                    uv = np.array([(x/self.size - 0.5) * 2, 
                                  (y/self.size - 0.5) * 2])
                    rd = rot_mat @ np.array([uv[0], uv[1], 1])
                    rd /= np.linalg.norm(rd)
                    self.pixels[x, y] = self.cast_ray(self.cam_pos, rd)
                
                if y % 10 == 0:
                    self.safe_update_image()

            self.safe_update_image()
            
        except Exception as e:
            print(f"Rendering error: {e}")
        finally:
            self.rendering = False
            if self.active:
                self.start_stop.config(text="Start")

    def safe_update_image(self):
        if not self.active:
            return
            
        try:
            self.tk_img = ImageTk.PhotoImage(image=self.img)
            self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)
        except tk.TclError:
            self.active = False

    def safe_reset(self):
        if not self.active:
            return
            
        self.rendering = False
        self.start_stop.config(text="Start")
        self.canvas.delete("all")
        self.initialize_fractal()
        if self.animating:
            self.start_animation()

    def cast_ray(self, ro, rd):
        d_total = 0
        for _ in range(50):
            p = ro + rd * d_total
            d = self.mandelbulb_distance(p)
            d_total += d
            if d < 0.001:
                normal = self.get_normal(p)
                light_dir = (self.light_pos - p)
                light_dir /= np.linalg.norm(light_dir)
                diff = max(0.1, np.dot(normal, light_dir))
                col = int(diff * 255)
                return (col, col, col)
            if d_total > 8:
                break
        return (0, 0, 0)
        
    def mandelbulb_distance(self, p):
        z = p.copy()
        dr = 1.0
        r = 0.0
        power = self.power
        
        for _ in range(self.iterations):
            r = np.linalg.norm(z)
            if r > 2:
                break
                
            theta = math.acos(z[2]/r)
            phi = math.atan2(z[1], z[0])
            dr = r**(power-1) * power * dr + 1.0
            
            zr = r**power
            theta *= power
            phi *= power
            
            z = zr * np.array([
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta)
            ]) + p
            
        return 0.5 * math.log(r) * r / dr if r > 2 else 9999.0
        
    def get_normal(self, p):
        eps = 0.001
        return np.array([
            self.mandelbulb_distance(p + [eps,0,0]) - self.mandelbulb_distance(p - [eps,0,0]),
            self.mandelbulb_distance(p + [0,eps,0]) - self.mandelbulb_distance(p - [0,eps,0]),
            self.mandelbulb_distance(p + [0,0,eps]) - self.mandelbulb_distance(p - [0,0,eps])
        ]) / (2*eps)

    def reset(self):
        self.safe_reset()

    def destroy(self):
        self.active = False
        self.rendering = False
        super().destroy()

class MandelbrotFrame(BaseFractal):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.xmin, self.xmax = -2.0, 1.0
        self.ymin, self.ymax = -1.5, 1.5

    def initialize_fractal(self):
        self.z = np.zeros((self.size, self.size), dtype=np.complex128)
        self.div_time = np.full((self.size, self.size), self.iterations, dtype=np.float32)
        self.current_iter = 0

    def start_animation(self):
        x = np.linspace(self.xmin, self.xmax, self.size)
        y = np.linspace(self.ymin, self.ymax, self.size)
        self.c = x[:, None] + 1j * y[None, :]
        self.animating = True
        self.animate()

    def animate(self):
        if not self.animating or self.current_iter >= self.iterations:
            return

        steps = 5
        for _ in range(steps):
            if self.current_iter >= self.iterations:
                break
            mask = (self.div_time == self.iterations)
            self.z[mask] = self.z[mask] ** 2 + self.c[mask]
            escaped = (np.abs(self.z) > 4) & mask
            self.div_time[escaped] = self.current_iter
            self.current_iter += 1

        img = Image.fromarray(np.uint8(self.div_time.T * 255 / self.iterations))
        self.tk_img = ImageTk.PhotoImage(image=img.convert('RGB'))
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)
        self.anim_id = self.after(self.animation_delay, self.animate)

class BurningShipFrame(BaseFractal):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.xmin, self.xmax = -2.0, 1.0
        self.ymin, self.ymax = -1.8, 0.2

    def initialize_fractal(self):
        self.z = np.zeros((self.size, self.size), dtype=np.complex128)
        self.div_time = np.full((self.size, self.size), self.iterations, dtype=np.float32)
        self.current_iter = 0

    def start_animation(self):
        x = np.linspace(self.xmin, self.xmax, self.size)
        y = np.linspace(self.ymin, self.ymax, self.size)
        self.c = x[:, None] + 1j * y[None, :]
        self.animating = True
        self.animate()

    def animate(self):
        if not self.animating or self.current_iter >= self.iterations:
            return

        steps = 5
        for _ in range(steps):
            if self.current_iter >= self.iterations:
                break
            mask = (self.div_time == self.iterations)
            self.z[mask] = (np.abs(self.z[mask].real) + 1j * np.abs(self.z[mask].imag)) ** 2 + self.c[mask]
            escaped = (np.abs(self.z) > 4) & mask
            self.div_time[escaped] = self.current_iter
            self.current_iter += 1

        img = Image.fromarray(np.uint8(self.div_time.T * 255 / self.iterations))
        self.tk_img = ImageTk.PhotoImage(image=img.convert('RGB'))
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)
        self.anim_id = self.after(self.animation_delay, self.animate)

class JuliaFrame(BaseFractal):
    def __init__(self, master, app, constant):
        self.constant = constant
        super().__init__(master, app)
        self.xmin, self.xmax = -2.0, 2.0
        self.ymin, self.ymax = -2.0, 2.0

    def initialize_fractal(self):
        self.z = np.zeros((self.size, self.size), dtype=np.complex128)
        self.div_time = np.full((self.size, self.size), self.iterations, dtype=np.float32)
        self.current_iter = 0

    def start_animation(self):
        x = np.linspace(self.xmin, self.xmax, self.size)
        y = np.linspace(self.ymin, self.ymax, self.size)
        self.z = x[:, None] + 1j * y[None, :]
        self.animating = True
        self.animate()

    def animate(self):
        if not self.animating or self.current_iter >= self.iterations:
            return

        steps = 5
        for _ in range(steps):
            if self.current_iter >= self.iterations:
                break
            mask = (self.div_time == self.iterations)
            self.z[mask] = self.z[mask] ** 2 + self.constant
            escaped = (np.abs(self.z) > 4) & mask
            self.div_time[escaped] = self.current_iter
            self.current_iter += 1

        img = Image.fromarray(np.uint8(self.div_time.T * 255 / self.iterations))
        self.tk_img = ImageTk.PhotoImage(image=img.convert('RGB'))
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)
        self.anim_id = self.after(self.animation_delay, self.animate)

class NewtonFrame(BaseFractal):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.xmin, self.xmax = -2.0, 2.0
        self.ymin, self.ymax = -2.0, 2.0

    def initialize_fractal(self):
        self.current_iter = 0

    def start_animation(self):
        x = np.linspace(self.xmin, self.xmax, self.size)
        y = np.linspace(self.ymin, self.ymax, self.size)
        X, Y = np.meshgrid(x, y)
        self.Z = X + Y * 1j
        self.animating = True
        self.animate()

    def animate(self):
        if not self.animating or self.current_iter >= self.iterations:
            return

        steps = 3
        for _ in range(steps):
            if self.current_iter >= self.iterations:
                break
            mask = np.abs(self.Z) > 1e-6
            self.Z[mask] -= (self.Z[mask] ** 3 - 1) / (3 * self.Z[mask] ** 2)
            self.current_iter += 1

        roots = [1, -0.5+0.866j, -0.5-0.866j]
        colors = np.array([[255,0,0], [0,255,0], [0,0,255], [100,100,100]])
        distances = np.abs(self.Z[:,:,None] - np.array(roots)[None,None,:])
        closest = np.argmin(distances, axis=2)
        img_array = colors[closest]

        self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(img_array.astype(np.uint8)))
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)
        self.anim_id = self.after(self.animation_delay, self.animate)

class SierpinskiFrame(BaseFractal):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.zoom_enabled = False
    def initialize_fractal(self):
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

    def start_animation(self):
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
        self.zoom_enabled = False
        
    def initialize_fractal(self):
        self.configure_transforms()
        self.img = Image.new('RGB', (self.size, self.size), 'black')
        self.pixels = self.img.load()
        self.x, self.y = 0.0, 0.0
        self.points_generated = 0

    def configure_transforms(self):
        if self.fractal_type == 'barnsley':
            self.transforms = [
                (0.85,0.04,-0.04,0.85,0,1.6,0.85),
                (0.2,-0.26,0.23,0.22,0,1.6,0.07),
                (-0.15,0.28,0.26,0.24,0,0.44,0.07),
                (0,0,0,0.16,0,0,0.01)
            ]
            self.color = (34,139,34)
            self.scale_x = self.size/11
            self.scale_y = -self.size/11
            self.offset_x = self.size/2
            self.offset_y = self.size

        elif self.fractal_type == 'koch':
            angle = math.pi / 3
            scale = 1 / 3
            sin60 = math.sin(angle)
            
            self.transforms = [
                (scale, 0, 0, scale, 0, 0, 0.25),
                (
                    scale * math.cos(angle),
                    -scale * math.sin(angle),
                    scale * math.sin(angle),
                    scale * math.cos(angle),
                    scale,
                    0,
                    0.25
                ),
                (
                    scale * math.cos(-angle),
                    -scale * math.sin(-angle),
                    scale * math.sin(-angle),
                    scale * math.cos(-angle),
                    0.5,
                    scale * sin60,
                    0.25
                ),
                (scale, 0, 0, scale, 2 * scale, 0, 0.25)
            ]
            
            self.color = (255, 255, 255)
            self.scale_x = self.size
            self.scale_y = -self.size
            self.offset_x = self.size * 0.1
            self.offset_y = self.size * 0.7
            self.points_generated = 0
        elif self.fractal_type == 'dragon':
            self.transforms = [
                (0.5,0.5,-0.5,0.5,0,0,0.5),
                (-0.5,-0.5,0.5,-0.5,1,0,0.5)
            ]
            self.color = (0,0,255)
            self.scale_x = self.size/3
            self.scale_y = self.size/3
            self.offset_x = self.size/2
            self.offset_y = self.size/2

        elif self.fractal_type == 'levy':
            self.transforms = [
                (0.5,-0.5,0.5,0.5,0,0,0.5),
                (0.5,0.5,-0.5,0.5,0.5,0.5,0.5)
            ]
            self.color = (255,0,0)
            self.scale_x = self.size/2
            self.scale_y = self.size/2
            self.offset_x = self.size/4
            self.offset_y = self.size/4

        elif self.fractal_type == 'tree':
            self.transforms = [
                (0.195,-0.488,0.344,0.443,0.4431,0.2452,0.4),
                (0.462,0.414,-0.252,0.361,0.2511,0.5692,0.4),
                (-0.058,-0.07,0.453,-0.111,0.5976,0.0969,0.15),
                (-0.035,0.07,-0.469,-0.022,0.488,0.5069,0.05)
            ]
            self.color = (0,200,0)
            self.scale_x = self.size/2.5
            self.scale_y = -self.size/2.5
            self.offset_x = self.size/2
            self.offset_y = self.size

    def start_animation(self):
        self.animating = True
        self.animate()

    def animate(self):
        if not self.animating or self.points_generated >= 20000:
            return

        batch_size = 500
        for _ in range(batch_size):
            if self.points_generated >= 20000:
                break
            
            r = random.random()
            cumulative = 0.0
            x, y = self.x, self.y
            
            for a, b, c, d, e, f, prob in self.transforms:
                cumulative += prob
                if r < cumulative:
                    x_new = a * x + b * y + e
                    y_new = c * x + d * y + f
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