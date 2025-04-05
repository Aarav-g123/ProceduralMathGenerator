import numpy as np
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import ttk
import math
import random
from scipy.signal import convolve2d

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

    def show_fractal(self, fractal_class):
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
        
        tk.Label(self, text="Fractal Explorer", font=("Arial", 24)).pack(pady=20)
        
        fractals = [
            ("Barnsley Fern", BarnsleyFernFrame),
            ("Burning Ship", BurningShipFrame),
            ("Game of Life", GameOfLifeFrame),
            ("Julia Set", JuliaFrame),
            ("Koch Snowflake", KochSnowflakeFrame),
            ("Mandelbrot", MandelbrotFrame),
            ("Newton", NewtonFrame),
            ("Sierpinski", SierpinskiFrame)
        ]
        
        for text, fractal_class in sorted(fractals, key=lambda x: x[0]):
            tk.Button(self, text=text, width=20, height=2,
                    command=lambda c=fractal_class: app.show_fractal(c)).pack(pady=2)
        
        tk.Button(self, text="Quit", width=20, height=2,
                command=master.destroy, bg="#ff9999").pack(pady=10)

class BaseFractalFrame(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.size = 600
        self.iterations = 50
        self.xmin, self.xmax = -2.0, 1.0
        self.ymin, self.ymax = -1.5, 1.5
        
        nav_frame = tk.Frame(self)
        nav_frame.pack(fill=tk.X)
        tk.Button(nav_frame, text="← Back", command=self.app.show_start_screen).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self, width=self.size, height=self.size)
        self.canvas.pack()

        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X)
        ttk.Button(control_frame, text="Reset", command=self.reset_view).pack(side=tk.LEFT, padx=5)
        ttk.Label(control_frame, text="Iterations:").pack(side=tk.LEFT)
        self.iter_slider = ttk.Scale(control_frame, from_=10, to=200, 
                                   command=lambda v: self.set_iterations(int(float(v))))
        self.iter_slider.set(self.iterations)
        self.iter_slider.pack(side=tk.LEFT, padx=5)

    def set_iterations(self, iterations):
        self.iterations = iterations
        self.draw_fractal()

    def reset_view(self):
        self.xmin, self.xmax = -2.0, 1.0
        self.ymin, self.ymax = -1.5, 1.5
        self.draw_fractal()

    def draw_fractal(self):
        pass

class NewtonFrame(BaseFractalFrame):
    def draw_fractal(self):
        x = np.linspace(-2, 2, self.size)
        y = np.linspace(-2, 2, self.size)
        X, Y = np.meshgrid(x, y)
        Z = X + Y*1j
        
        for _ in range(self.iterations):
            mask = np.abs(Z) > 1e-6
            Z[mask] -= (Z[mask]**3 - 1) / (3*Z[mask]**2)
        
        roots = [1, -0.5+0.866j, -0.5-0.866j]
        colors = np.array([[255,0,0], [0,255,0], [0,0,255], [100,100,100]])
        distances = np.abs(Z[:,:,None] - np.array(roots)[None,None,:])
        closest = np.argmin(distances, axis=2)
        img_array = colors[closest]
        
        self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(img_array.astype(np.uint8)))
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

class GameOfLifeFrame(BaseFractalFrame):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.grid_size = 100
        self.cell_size = self.size // self.grid_size
        self.grid = np.random.choice([0,1], (self.grid_size, self.grid_size))
        self.running = False
        self.iter_slider.pack_forget()
        
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X)
        ttk.Button(control_frame, text="Start", command=self.start_life).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop", command=self.stop_life).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Reset", command=self.reset_life).pack(side=tk.LEFT, padx=5)

    def start_life(self):
        if not self.running:
            self.running = True
            self.update_life()

    def stop_life(self):
        self.running = False

    def reset_life(self):
        self.grid = np.random.choice([0,1], (self.grid_size, self.grid_size))
        self.draw_fractal()

    def update_life(self):
        if self.running:
            self.draw_fractal()
            self.after(50, self.update_life)

    def draw_fractal(self):
        kernel = np.array([[1,1,1], [1,0,1], [1,1,1]])
        neighbors = convolve2d(self.grid, kernel, mode='same', boundary='wrap')
        birth = (neighbors == 3) & (self.grid == 0)
        survive = ((neighbors == 2) | (neighbors == 3)) & (self.grid == 1)
        self.grid = np.where(birth | survive, 1, 0)
        
        img = Image.new('RGB', (self.size, self.size))
        img_array = np.repeat(self.grid[:,:,None], 3, axis=2) * 255
        img.paste(Image.fromarray(img_array.astype(np.uint8)).resize((self.size, self.size), Image.NEAREST))
        self.tk_img = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

class MandelbrotFrame(BaseFractalFrame):
    def draw_fractal(self):
        x = np.linspace(self.xmin, self.xmax, self.size)
        y = np.linspace(self.ymin, self.ymax, self.size)
        c = x[:,None] + 1j*y[None,:]
        z = np.zeros_like(c)
        div_time = np.full(c.shape, self.iterations, dtype=np.float32)
        
        for i in range(self.iterations):
            mask = (div_time == self.iterations)
            z[mask] = z[mask]**2 + c[mask]
            div_time[(np.abs(z) > 2) & mask] = i
        
        hue = (div_time.T * 255 / self.iterations).astype(np.uint8)
        self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(hue).convert('RGB'))
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

class JuliaFrame(MandelbrotFrame):
    def __init__(self, master, app):
        self.c = complex(-0.4, 0.6)
        super().__init__(master, app)

    def draw_fractal(self):
        x = np.linspace(self.xmin, self.xmax, self.size)
        y = np.linspace(self.ymin, self.ymax, self.size)
        z = x[:,None] + 1j*y[None,:]
        div_time = np.full(z.shape, self.iterations, dtype=np.float32)
        
        for i in range(self.iterations):
            mask = (div_time == self.iterations)
            z[mask] = z[mask]**2 + self.c
            div_time[(np.abs(z) > 2) & mask] = i
        
        hue = (div_time.T * 255 / self.iterations).astype(np.uint8)
        self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(hue).convert('RGB'))
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

class BurningShipFrame(MandelbrotFrame):
    def draw_fractal(self):
        x = np.linspace(self.xmin, self.xmax, self.size)
        y = np.linspace(self.ymin, self.ymax, self.size)
        c = x[:,None] + 1j*y[None,:]
        z = np.zeros_like(c)
        div_time = np.full(c.shape, self.iterations, dtype=np.float32)
        
        for i in range(self.iterations):
            mask = (div_time == self.iterations)
            z[mask] = (np.abs(z[mask].real) + 1j*np.abs(z[mask].imag))**2 + c[mask]
            div_time[(np.abs(z) > 2) & mask] = i
        
        hue = (div_time.T * 255 / self.iterations).astype(np.uint8)
        self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(hue).convert('RGB'))
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

class SierpinskiFrame(BaseFractalFrame):
    def draw_fractal(self):
        img = Image.new('RGB', (self.size, self.size), 'black')
        draw = ImageDraw.Draw(img)
        
        def draw_triangle(points, depth):
            if depth == 0:
                draw.polygon(points, fill='white')
                return
            
            mid1 = (points[0][0] + points[1][0])/2, (points[0][1] + points[1][1])/2
            mid2 = (points[1][0] + points[2][0])/2, (points[1][1] + points[2][1])/2
            mid3 = (points[2][0] + points[0][0])/2, (points[2][1] + points[0][1])/2
            
            draw_triangle([points[0], mid1, mid3], depth-1)
            draw_triangle([mid1, points[1], mid2], depth-1)
            draw_triangle([mid3, mid2, points[2]], depth-1)
        
        margin = 50
        start_points = [
            (self.size//2, margin),
            (margin, self.size - margin),
            (self.size - margin, self.size - margin)
        ]
        
        draw_triangle(start_points, 6)
        self.tk_img = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

class BarnsleyFernFrame(BaseFractalFrame):
    def draw_fractal(self):
        transforms = [
            (0.85, 0.04, -0.04, 0.85, 0, 1.6),
            (0.2, -0.26, 0.23, 0.22, 0, 1.6),
            (-0.15, 0.28, 0.26, 0.24, 0, 0.44),
            (0, 0, 0, 0.16, 0, 0)
        ]
        probs = [0.85, 0.07, 0.07, 0.01]
        
        x, y = 0, 0
        points = np.zeros((20000, 2))
        for i in range(len(points)):
            r = np.random.rand()
            idx = np.searchsorted(np.cumsum(probs), r)
            a,b,c,d,e,f = transforms[idx]
            x, y = a*x + b*y + e, c*x + d*y + f
            points[i] = [x, y]
        
        points = (points * [self.size/11, -self.size/11] + [self.size/2, self.size]).astype(int)
        valid = (points[:,0] >= 0) & (points[:,0] < self.size) & (points[:,1] >= 0) & (points[:,1] < self.size)
        img = Image.new('RGB', (self.size, self.size))
        pixels = img.load()
        np.add.at(pixels, tuple(points[valid].T), (34, 139, 34))
        self.tk_img = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

class KochSnowflakeFrame(BaseFractalFrame):
    def draw_fractal(self):
        depth = 5
        size = self.size * 0.8
        offset = self.size * 0.1
        points = np.array([[offset, size], [size, size], [size/2 + offset, size - size*math.sqrt(3)/2]])
        img = Image.new('RGB', (self.size, self.size), 'black')
        draw = ImageDraw.Draw(img)
        
        def koch(p1, p2, level):
            if level == 0:
                draw.line([tuple(p1), tuple(p2)], fill='white', width=2)
                return
            diff = p2 - p1
            p3 = p1 + diff/3
            p4 = p2 - diff/3
            angle = np.arctan2(diff[1], diff[0]) + np.pi/3
            length = np.hypot(diff[0], diff[1])/3
            p5 = p3 + np.array([length*np.cos(angle), length*np.sin(angle)])
            
            koch(p1, p3, level-1)
            koch(p3, p5, level-1)
            koch(p5, p4, level-1)
            koch(p4, p2, level-1)
        
        for i in range(3):
            koch(points[i], points[(i+1)%3], depth)
        
        self.tk_img = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

if __name__ == "__main__":
    root = tk.Tk()
    app = FractalApp(root)
    root.mainloop()