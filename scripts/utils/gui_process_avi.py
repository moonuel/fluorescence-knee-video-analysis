"""
GUI wrapper for the AVI to centered NPY processing script.
Provides file selection interface with automatic filename generation.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys
import subprocess
import threading


class AVIProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AVI to Centered NPY Converter")
        self.root.geometry("600x200")
        self.root.resizable(True, False)

        # Variables
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Input file section
        input_frame = tk.Frame(self.root)
        input_frame.pack(pady=10, padx=20, fill=tk.X)

        tk.Label(input_frame, text="Input AVI File:").grid(row=0, column=0, sticky=tk.W)
        self.input_entry = tk.Entry(input_frame, textvariable=self.input_file, width=50)
        self.input_entry.grid(row=1, column=0, padx=(0, 10))
        tk.Button(input_frame, text="Browse...", command=self.browse_input).grid(row=1, column=1)

        # Output file section
        output_frame = tk.Frame(self.root)
        output_frame.pack(pady=10, padx=20, fill=tk.X)

        tk.Label(output_frame, text="Output NPY File:").grid(row=0, column=0, sticky=tk.W)
        self.output_entry = tk.Entry(output_frame, textvariable=self.output_file, width=50)
        self.output_entry.grid(row=1, column=0, padx=(0, 10))
        tk.Button(output_frame, text="Browse...", command=self.browse_output_file).grid(row=1, column=1)

        # Process button
        self.process_button = tk.Button(self.root, text="Process Video", command=self.process_video,
                                       bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
        self.process_button.pack(pady=20)

        # Status label
        self.status_label = tk.Label(self.root, text="", fg="blue")
        self.status_label.pack()

    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Select AVI file",
            filetypes=[("AVI files", "*.avi"), ("All files", "*.*")]
        )
        if filename:
            self.input_file.set(filename)
            self.auto_fill_output(filename)

    def browse_output_file(self):
        current_output = self.output_file.get()
        initial_dir = os.path.dirname(current_output) if current_output else None
        initial_file = os.path.basename(current_output) if current_output else None

        filename = filedialog.asksaveasfilename(
            title="Save NPY file as",
            initialdir=initial_dir,
            initialfile=initial_file,
            filetypes=[("NPY files", "*.npy"), ("All files", "*.*")]
        )
        if filename:
            # Ensure .npy extension
            if not filename.lower().endswith('.npy'):
                filename += '.npy'
            self.output_file.set(filename)

    def auto_fill_output(self, input_path):
        if input_path:
            input_dir = os.path.dirname(input_path)
            input_basename = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(input_dir, f"{input_basename}.npy")
            self.output_file.set(output_path)

    def process_video(self):
        input_path = self.input_file.get()
        output_path = self.output_file.get()

        if not input_path:
            messagebox.showerror("Error", "Please select an input AVI file.")
            return
        if not output_path:
            messagebox.showerror("Error", "Please specify an output NPY file.")
            return

        if not os.path.exists(input_path):
            messagebox.showerror("Error", "Input file does not exist.")
            return

        # Disable button and show processing
        self.process_button.config(state=tk.DISABLED, text="Processing...")
        self.status_label.config(text="Processing video... Please wait.", fg="orange")

        # Run processing in separate thread to avoid freezing GUI
        thread = threading.Thread(target=self.run_processing, args=(input_path, output_path))
        thread.daemon = True
        thread.start()

    def run_processing(self, input_path, output_path):
        try:
            # Import the processing script from the same directory
            from process_avi_to_centered_npy import main

            # Run the processing
            main(input_path, output_path)

            # Success
            self.root.after(0, lambda: self.status_label.config(text="Processing completed successfully!", fg="green"))
            self.root.after(0, lambda: messagebox.showinfo("Success", "Video processing completed successfully!"))

        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            self.root.after(0, lambda: self.status_label.config(text=error_msg, fg="red"))
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))

        finally:
            # Re-enable button
            self.root.after(0, lambda: self.process_button.config(state=tk.NORMAL, text="Process Video"))


def main():
    root = tk.Tk()
    app = AVIProcessorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
