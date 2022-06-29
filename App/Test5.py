import tkinter as tk
import numpy as np
from PIL import ImageTk, Image
import os

class StatePosition(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        #print(os.getcwd())
        self.pos = None
        self.hold = None
        self.label = tk.Label(self, text = "State Number")
        self.entry = tk.Entry(self)
        self.random_start = tk.Button(self, text="Random Start", command=self.random)
        self.button = tk.Button(self, text="Click to Enter State", command=self.on_button)
        self.img = ImageTk.PhotoImage(Image.open("Labyrinth_final.png"))
        self.pic = tk.Label(self, image = self.img)
        self.button.pack()
        self.button.place(x = 40, y = 100)
        self.entry.pack(side = "bottom")
        self.entry.place(x =40, y = 60)
        self.random_start.pack()
        self.random_start.place(x = 40, y = 140)
        self.label.pack()
        self.label.place(x= 40, y = 40)
        self.geometry("800x800")
        self.pic.pack(side = "top")
        self.alph_pos = ['a2','a3','a4','a5','a6','a10','a11','a12','a13','b2','b6','b10','c2','c4','c6','c7','c8','c9','c10','c11','c12','d2','d4','d9','d14','e1','e2','e3','e4','e5','e6','e11','e12','e13','e14','f1','f6','f11','g1','g2','g3','g4','g6','g7','g8','g9','g10','g11','g12','g13','g14','h6','h11','h14','i1','i2','i3','i4','i5','i6','i7','i8','i9','i10','i11','i12','i13','i14','j1','j6','j10','j14','k1','k3','k6','k10','l1','l3','l4','l5','l6','l7','l9','l10','l11','l12','l13','l14']
        self.num_pos = [44, 46, 48, 50, 52, 60, 62, 64, 66, 84, 92, 100, 124, 128, 132, 134, 136, 138, 140, 142, 144, 164, 168, 178, 188, 202, 204, 206, 208, 210, 212, 222, 224, 226, 228, 242, 252, 262, 282, 284, 286, 288, 292, 294, 296, 298, 300, 302, 304, 306, 308, 332, 342, 348, 362, 364, 366, 368, 370, 372, 374, 376, 378, 380, 382, 384, 386, 388, 402, 412, 420, 428, 442, 446, 452, 460, 482, 486, 488, 490, 492, 494, 498, 500, 502, 504, 506, 508]
        #print(len(self.alph_pos))
        #print(len(self.num_pos))

    def on_button(self):
        num = self.entry.get().lower()
        if num in self.alph_pos:
            index = self.alph_pos.index(num)
            self.pos = self.num_pos[index]
            self.quit()

    def random(self):
        self.pos = np.random.choice(self.num_pos)
        self.quit()

#Hearing()
def state_pos():
    promptData = StatePosition()
    promptData.mainloop()
    promptData.destroy()
    pos = promptData.pos
    return pos

if __name__ == "__main__":
    r= state_pos()
    print(r)
    # app = StatePosition()
    # app.mainloop()
    # #print(app.on_button())