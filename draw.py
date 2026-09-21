import tkinter as tk
import torch
from model import load_model

class DigitDrawApp:
    def __init__(self, root, model_path="mnist_mlp_best.pt"):
        self.root = root
        self.root.title("Draw 28x28 -> MNIST Predictor")

        self.model, self.device = load_model(model_path)

        #this grid is the model input (28x28)
        self.grid = torch.zeros((28, 28), dtype=torch.float32)

        self.cell = 16   # size of each pixel cell on screen
        self.brush = 1   # brush radius in cells (1 => 3x3)


        self.canvas = tk.Canvas(root, width=28*self.cell, height=28*self.cell, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        self.rects = [[None]*28 for _ in range(28)]
        for r in range(28):
            for c in range(28):
                x1, y1 = c*self.cell, r*self.cell
                x2, y2 = x1 + self.cell, y1 + self.cell
                self.rects[r][c] = self.canvas.create_rectangle(
                    x1, y1, x2, y2, outline="#e5e5e5", fill="white"
                )

        self.canvas.bind("<Button-1>", self.on_draw)
        self.canvas.bind("<B1-Motion>", self.on_draw)

        self.pred_label = tk.Label(root, text="Draw a digit, then click Predict", font=("Arial", 14))
        self.pred_label.grid(row=1, column=0, columnspan=4, pady=(0, 10))

        tk.Button(root, text="Predict", width=12, command=self.predict).grid(row=2, column=0, padx=5, pady=5)
        tk.Button(root, text="Clear", width=12, command=self.clear).grid(row=2, column=1, padx=5, pady=5)

        tk.Label(root, text="Brush:").grid(row=2, column=2, sticky="e")
        self.brush_var = tk.IntVar(value=self.brush)
        tk.Spinbox(root, from_=0, to=3, width=5, textvariable=self.brush_var,
                   command=self.set_brush).grid(row=2, column=3, sticky="w")

        tk.Label(root, text="Tip: brush 1–2 works best.", fg="gray").grid(row=3, column=0, columnspan=4)

    def set_brush(self):
        self.brush = int(self.brush_var.get())

    def on_draw(self, event):
        c = event.x // self.cell
        r = event.y // self.cell
        if not (0 <= r < 28 and 0 <= c < 28):
            return

        self.set_brush()
        b = self.brush
        for rr in range(max(0, r-b), min(28, r+b+1)):
            for cc in range(max(0, c-b), min(28, c+b+1)):
                self.grid[rr, cc] = 1.0
                self.canvas.itemconfig(self.rects[rr][cc], fill="black")

    def clear(self):
        self.grid.zero_()
        for r in range(28):
            for c in range(28):
                self.canvas.itemconfig(self.rects[r][c], fill="white")
        self.pred_label.config(text="Cleared — draw again")

    @torch.no_grad()
    def predict(self):
        x = self.grid.unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,28,28)

        probs = torch.softmax(self.model(x), dim=1).squeeze(0)
        pred = int(probs.argmax().item())
        conf = float(probs[pred].item())

        top3 = torch.topk(probs, k=3)
        top3_str = ", ".join([f"{int(i)}:{float(v):.3f}" for v, i in zip(top3.values, top3.indices)])

        self.pred_label.config(text=f"Prediction: {pred} (conf {conf:.3f}) | Top-3: {top3_str}")

def main():
    root = tk.Tk()
    DigitDrawApp(root, model_path="mnist_mlp_best.pt")
    root.mainloop()

if __name__ == "__main__":
    main()