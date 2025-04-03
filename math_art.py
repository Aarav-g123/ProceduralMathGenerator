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
        self.iterations = 100
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
        self.iter_slider = ttk.Scale(control_frame, from_=50, to=500, 
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
        img = Image.new('RGB', (self.size, self.size))
        pixels = img.load()
        
        for x in range(self.size):
            for y in range(self.size):
                z = complex((x - self.size/2)/(self.size/4), (y - self.size/2)/(self.size/4))
                for _ in range(self.iterations):
                    try:
                        z = z - (z**3 - 1)/(3*z**2)
                    except:
                        z = 0
                        break
                
                if z != 0:
                    root = (round(z.real,2), round(z.imag,2))
                    colors = {(1.0,0.0):(255,0,0), (-0.5,0.87):(0,255,0), (-0.5,-0.87):(0,0,255)}
                    color = colors.get(root, (100,100,100))
                else:
                    color = (0,0,0)
                
                pixels[x,y] = color
        
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

class GameOfLifeFrame(BaseFractalFrame):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.grid_size = 100
        self.cell_size = self.size // self.grid_size
        self.grid = np.random.choice([0,1], (self.grid_size,self.grid_size))
        self.running = False
        self.iter_slider.pack_forget()
        
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X)
        ttk.Button(control_frame, text="Start", command=self.start_life).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop", command=self.stop_life).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Reset", command=self.reset_life).pack(side=tk.LEFT, padx=5)

    def start_life(self):
        self.running = True
        self.update_life()

    def stop_life(self):
        self.running = False

    def reset_life(self):
        self.grid = np.random.choice([0,1], (self.grid_size,self.grid_size))
        self.draw_fractal()

    def update_life(self):
        if self.running:
            self.draw_fractal()
            self.after(100, self.update_life)

    def draw_fractal(self):
        new_grid = np.copy(self.grid)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                neighbors = np.sum(self.grid[max(0,i-1):min(self.grid_size,i+2),
                                      max(0,j-1):min(self.grid_size,j+2)]) - self.grid[i,j]
                if self.grid[i,j] == 1 and (neighbors < 2 or neighbors > 3):
                    new_grid[i,j] = 0
                elif self.grid[i,j] == 0 and neighbors == 3:
                    new_grid[i,j] = 1
        self.grid = new_grid
        
        img = Image.new('RGB', (self.size, self.size))
        draw = ImageDraw.Draw(img)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i,j]:
                    x = i * self.cell_size
                    y = j * self.cell_size
                    draw.rectangle([x,y,x+self.cell_size,y+self.cell_size], fill=(255,255,255))
        
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

class MandelbrotFrame(BaseFractalFrame):
    def draw_fractal(self):
        x = np.linspace(self.xmin, self.xmax, self.size)
        y = np.linspace(self.ymin, self.ymax, self.size)
        c = x[:,None] + 1j*y[None,:]
        z = np.zeros_like(c)
        div_time = np.zeros(c.shape, int)
        
        for i in range(self.iterations):
            mask = (div_time == 0)
            z[mask] = z[mask]**2 + c[mask]
            div_time[(np.abs(z) >= 2) & (div_time == 0)] = i
        
        hue = (div_time.T * 255 // self.iterations).astype(np.uint8)
        img = Image.fromarray(hue, 'L').convert('RGB')
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

class JuliaFrame(BaseFractalFrame):
    def __init__(self, master, app):
        self.c = complex(-0.4, 0.6)
        super().__init__(master, app)

    def draw_fractal(self):
        x = np.linspace(self.xmin, self.xmax, self.size)
        y = np.linspace(self.ymin, self.ymax, self.size)
        z = x[:,None] + 1j*y[None,:]
        div_time = np.zeros(z.shape, int)
        
        for i in range(self.iterations):
            mask = (div_time == 0)
            z[mask] = z[mask]**2 + self.c
            div_time[(np.abs(z) >= 2) & (div_time == 0)] = i
        
        hue = (div_time.T * 255 // self.iterations).astype(np.uint8)
        img = Image.fromarray(hue, 'L').convert('RGB')
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

class BurningShipFrame(BaseFractalFrame):
    def draw_fractal(self):
        x = np.linspace(self.xmin, self.xmax, self.size)
        y = np.linspace(self.ymin, self.ymax, self.size)
        c = x[:,None] + 1j*y[None,:]
        z = np.zeros_like(c)
        div_time = np.zeros(c.shape, int)
        
        for i in range(self.iterations):
            mask = (div_time == 0)
            z[mask] = (np.abs(z[mask].real) + 1j*np.abs(z[mask].imag))**2 + c[mask]
            div_time[(np.abs(z) >= 2) & (div_time == 0)] = i
        
        hue = (div_time.T * 255 // self.iterations).astype(np.uint8)
        img = Image.fromarray(hue, 'L').convert('RGB')
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

class SierpinskiFrame(BaseFractalFrame):
    def draw_fractal(self):
        img = Image.new('RGB', (self.size, self.size), 'black')
        draw = ImageDraw.Draw(img)
        
        def draw_triangle(points, depth=6):
            if depth == 0:
                draw.polygon(points, fill='white')
                return
            
            mid1 = ((points[0][0]+points[1][0])/2, (points[0][1]+points[1][1])/2)
            mid2 = ((points[1][0]+points[2][0])/2, (points[1][1]+points[2][1])/2)
            mid3 = ((points[2][0]+points[0][0])/2, (points[2][1]+points[0][1])/2)
            
            draw_triangle([points[0], mid1, mid3], depth-1)
            draw_triangle([mid1, points[1], mid2], depth-1)
            draw_triangle([mid3, mid2, points[2]], depth-1)
        
        margin = 50
        start_points = [
            (self.size//2, margin),
            (margin, self.size-margin),
            (self.size-margin, self.size-margin)
        ]
        
        draw_triangle(start_points)
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

class BarnsleyFernFrame(BaseFractalFrame):
    def draw_fractal(self):
        img = Image.new('RGB', (self.size, self.size))
        draw = ImageDraw.Draw(img)
        x, y = 0, 0
        
        for _ in range(100000):
            r = random.random()
            if r < 0.01:
                x, y = 0, 0.16*y
            elif r < 0.86:
                x, y = 0.85*x + 0.04*y, -0.04*x + 0.85*y + 1.6
            elif r < 0.93:
                x, y = 0.2*x - 0.26*y, 0.23*x + 0.22*y + 1.6
            else:
                x, y = -0.15*x + 0.28*y, 0.26*x + 0.24*y + 0.44
            
            ix = int(self.size/2 + x * self.size/10)
            iy = int(self.size - y * self.size/10)
            if 0 <= ix < self.size and 0 <= iy < self.size:
                draw.point((ix, iy), fill=(34, 139, 34))
        
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

class KochSnowflakeFrame(BaseFractalFrame):
    def draw_fractal(self):
        img = Image.new('RGB', (self.size, self.size), 'black')
        draw = ImageDraw.Draw(img)
        
        def koch(p1, p2, depth=4):
            if depth == 0:
                draw.line([p1, p2], fill='white', width=2)
                return
            
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            p3 = (p1[0] + dx/3, p1[1] + dy/3)
            p4 = (p2[0] - dx/3, p2[1] - dy/3)
            angle = math.atan2(dy, dx) + math.pi/3
            length = math.hypot(dx, dy)/3
            p5 = (p3[0] + length*math.cos(angle), p3[1] + length*math.sin(angle))
            
            koch(p1, p3, depth-1)
            koch(p3, p5, depth-1)
            koch(p5, p4, depth-1)
            koch(p4, p2, depth-1)
        
        size = self.size * 0.8
        offset = self.size * 0.1
        start_points = [
            (offset, size),
            (size, size),
            (size/2 + offset, size - size*math.sqrt(3)/2)
        ]
        for i in range(3):
            koch(start_points[i], start_points[(i+1)%3])
        
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

if __name__ == "__main__":
    root = tk.Tk()
    app = FractalApp(root)
    root.mainloop()