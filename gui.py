import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import agent

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.grid()
        self.create_widgets()
        self.ticker = "MSFT"
        self.running = False

    def create_widgets(self):
        self.ticker_label = tk.Label(self, text="Ticker:")
        self.ticker_label.grid(row=0, column=0)

        self.ticker_entry = tk.Entry(self)
        self.ticker_entry.insert(0, "MSFT")
        self.ticker_entry.grid(row=0, column=1)

        self.start_button = tk.Button(self, text="Start", command=self.start)
        self.start_button.grid(row=1, column=0)

        self.stop_button = tk.Button(self, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_button.grid(row=1, column=1)

    def start(self):
        self.ticker = self.ticker_entry.get()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.running = True
        self.ani = animation.FuncAnimation(plt.gcf(), self.update, interval=1000)
        plt.show()

    def stop(self):
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.running = False
        self.ani.event_source.stop()

    def update(self, i):
        if self.running:
            agent.train_and_predict(self.ticker)
            plt.clf()
            plt.plot(agent.live_prices, label="Live Data")
            plt.plot(agent.predictions, label="Predictions")
            plt.legend()

root = tk.Tk()
app = Application(master=root)
app.mainloop()
