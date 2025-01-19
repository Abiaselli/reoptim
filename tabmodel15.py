import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import json
import threading
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, AddedToken
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers
import psutil
from torch.autograd import Function


max_len = 128

class TensorCellGUI:
    def __init__(self, root):

        self.num_heads = tk.IntVar(value=16)
        self.layers = []

        # Model Configuration Variables
        self.model_name = tk.StringVar(value="Cellformer")
        self.num_parameters = tk.IntVar(value=128)

        self.vocab_size = tk.IntVar(value=30000)
        self.hidden_size = tk.IntVar(value=24)
        self.num_heads = tk.IntVar(value=3)
        self.num_layers = tk.IntVar(value=4)

        self.pad_token_id = 1  # Default value, adjust based on your tokenizer setup

        # Device Selection Variable
        self.device_option = tk.StringVar(value="GPU" if torch.cuda.is_available() else "CPU")
        
        # Training Parameters
        self.dataset_path = ""
        self.vocab_path = ""
        self.tokenizer_path = ""
        self.batch_size = tk.IntVar(value=1)
        self.learning_rate = tk.DoubleVar(value=0.0002)
        self.epochs = tk.IntVar(value=1)

        # Training Variables
        self.loss_history = []
        self.accuracy_history = []
        self.current_epoch = 0
        self.stop_training = threading.Event()
        
        # Model and Data Variables
        self.model = None
        self.tokenizer = None
        self.dataset_path = None
        self.tokenizer_path = None
        self.model_path = None
        self.train_data = None  # To store the dataset
        self.tokenized_data_path = None  # To store the tokenized data file path

        # Device (CPU or GPU) - Initially set based on device_option
        self.device = torch.device(self.map_device(self.device_option.get()))

        self.root = root
        self.root.title("3D Tensor Cell-Based GUI with Tensors")
        self.root.geometry("1400x800")

        # Main layout with a horizontal paned window
        self.paned_window = tk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill="both", expand=True)

        # Frame for the scrollable notebook
        notebook_frame = tk.Frame(self.paned_window)
        self.paned_window.add(notebook_frame, stretch="always")

        # Canvas and scrollbars for the notebook
        self.notebook_canvas = tk.Canvas(notebook_frame)
        self.notebook_canvas.pack(side="left", fill="both", expand=True)

        self.v_scrollbar = ttk.Scrollbar(notebook_frame, orient="vertical", command=self.notebook_canvas.yview)
        self.v_scrollbar.pack(side="right", fill="y")

        self.h_scrollbar = ttk.Scrollbar(notebook_frame, orient="horizontal", command=self.notebook_canvas.xview)
        self.h_scrollbar.pack(side="bottom", fill="x")

        # Configure the canvas to use the scrollbars
        self.notebook_canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        # Frame inside the canvas to hold the notebook
        self.notebook_frame = tk.Frame(self.notebook_canvas)
        self.notebook_canvas.create_window((0, 0), window=self.notebook_frame, anchor="nw")

        # Track changes to the notebook's size to adjust the scroll region
        self.notebook_frame.bind("<Configure>", lambda e: self.notebook_canvas.configure(scrollregion=self.notebook_canvas.bbox("all")))

        # Notebook for Z-axis tabs
        self.notebook = ttk.Notebook(self.notebook_frame)
        self.notebook.pack(fill="both", expand=True)

        # Add additional notebook for preparation tabs
        self.extra_notebook = ttk.Notebook(self.paned_window)
        self.paned_window.add(self.extra_notebook, minsize=200)


        # Set the resizing weights (e.g., the side panel gets a higher priority in resizing)
        self.paned_window.paneconfigure(self.extra_notebook, stretch="never")  # Prevents resizing
        # Add initial tab (Z=1)
        self.grid_data = {}  # To store cell data across tabs
        self.add_tab()

        # Menu for operations
        self.menu = tk.Menu(root)
        root.config(menu=self.menu)
        self.menu.add_command(label="Add Tab (Z+1)", command=self.add_tab)
        self.menu.add_command(label="Perform Tensor Operations", command=self.perform_tensor_operations)
        self.menu.add_command(label="Save Grid", command=self.save_grid)
        self.menu.add_command(label="Load Grid", command=self.load_grid)

        self.add_cell_details_tab() 
        # training and model preparation tabs
        self.add_training_and_model_tabs()
        # training controls at the bottom
        self.add_training_controls()
        self.add_numerical_parameters_tab()

        
        self.selected_cell = None  # Track selected cell
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)



        # Select log file path
        self.select_log_file()

        # Setup logging
        logging.basicConfig(filename=self.log_file_path, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        logging.info(f"Using device: {self.device}")


    def map_device(self, selected_device):
        device_mapping = {
            "CPU": "cpu",
            "GPU": "cuda"
        }
        return device_mapping.get(selected_device, "cpu")

    def add_tab(self):
        """Add a new tab to the scrollable notebook."""
        z_index = len(self.notebook.tabs()) + 1
        frame = tk.Frame(self.notebook)
        self.notebook.add(frame, text=f"Z={z_index}")

        # Initialize the grid for this tab
        self.grid_data[z_index] = {
            "grid": ttk.Frame(frame),
            "cells": {},
            "rows": 5,
            "columns": 5,
            "operation": None if z_index % 2 == 1 else "Add",
        }

        self.grid_data[z_index]["grid"].pack(fill="both", expand=True)
        self.initialize_grid(z_index)


    def on_tab_changed(self, event):
        """Handle tab change events."""
        selected_tab = self.notebook.index(self.notebook.select()) + 1  # Tabs are 1-indexed
        self.update_sidebar(selected_tab)

    def update_sidebar(self, z_index):
        """Update the sidebar based on the selected tab."""
        if z_index % 2 == 0:  # Even tab
            self.operation_label.pack(pady=5)
            self.operation_menu.pack()
        else:  # Odd tab
            self.operation_label.pack_forget()
            self.operation_menu.pack_forget()

    def add_tab(self):
        """Add a new tab to the notebook."""
        z_index = len(self.notebook.tabs()) + 1
        frame = tk.Frame(self.notebook)
        self.notebook.add(frame, text=f"Z={z_index}")

        # Initialize the grid for this tab
        self.grid_data[z_index] = {
            "grid": ttk.Frame(frame),
            "cells": {},
            "rows": 5,
            "columns": 5,
            "operation": None if z_index % 2 == 1 else "Add",
        }

        self.grid_data[z_index]["grid"].pack(fill="both", expand=True)
        self.initialize_grid(z_index)


    def add_x_axis(self):
        """Add a column (X-axis) to all tabs."""
        for z_index, data in self.grid_data.items():
            # Increment columns count
            data["columns"] += 1
            grid = data["grid"]

            # Add new column header
            col_label = ttk.Label(grid, text=f"X{data['columns']}", anchor="center", relief="raised")
            col_label.grid(row=0, column=data["columns"], sticky="nsew")

            # Add new cells for the column
            for row in range(1, data["rows"] + 1):
                self.create_cell(z_index, row, data["columns"])

    def add_y_axis(self):
        """Add a row (Y-axis) to all tabs."""
        for z_index, data in self.grid_data.items():
            # Increment rows count
            data["rows"] += 1
            grid = data["grid"]

            # Add new row header
            row_label = ttk.Label(grid, text=f"Y{data['rows']}", anchor="center", relief="raised")
            row_label.grid(row=data["rows"], column=0, sticky="nsew")

            # Add new cells for the row
            for col in range(1, data["columns"] + 1):
                self.create_cell(z_index, data["rows"], col)


    def initialize_grid(self, z_index):
        """Initialize the grid for the given Z-layer."""
        data = self.grid_data[z_index]
        grid = data["grid"]

        # Add labels for X-axis
        for col in range(data["columns"]):
            label = ttk.Label(grid, text=f"X{col+1}", anchor="center", relief="raised")
            label.grid(row=0, column=col+1, sticky="nsew")

        # Add labels for Y-axis and cells
        for row in range(1, data["rows"] + 1):
            label = ttk.Label(grid, text=f"Y{row}", anchor="center", relief="raised")
            label.grid(row=row, column=0, sticky="nsew")

            for col in range(1, data["columns"] + 1):
                self.create_cell(z_index, row, col)

    def clear_selected_cell(self):
        """Clear the selected cell."""
        if not self.selected_cell:
            messagebox.showinfo("No Cell Selected", "Please select a cell to clear.")
            return

        confirm = messagebox.askyesno("Confirm Clear", "Are you sure you want to clear the selected cell?")
        if confirm:
            cell_id = self.selected_cell["id"]
            z_index = int(cell_id.split(":")[0][1:])
            row = int(cell_id.split(":")[2][1:])
            col = int(cell_id.split(":")[1][1:])
            cell = self.grid_data[z_index]["cells"].get((row, col))

            if cell:
                # Clear the cell
                cell["entry"].delete(0, tk.END)
                cell["tensor"] = None
                self.select_cell(cell_id, cell["entry"])  # Update the sidebar


    def clear_all_cells(self):
        """Clear the active layer."""
        confirm = messagebox.askyesno("Confirm Clear All", "Are you sure you want to clear all cells?")
        if confirm:
            
            active_tab_index = self.notebook.index(self.notebook.select()) + 1  # Tabs are 1-indexed
            if active_tab_index in self.grid_data:
                data = self.grid_data[active_tab_index]
                for (row, col), cell in data["cells"].items():
                    cell["entry"].delete(0, tk.END)
                    cell["tensor"] = None
                messagebox.showinfo("Grid Cleared", f"Layer Z={active_tab_index} has been cleared.")
            else:
                messagebox.showinfo("No Active Tab", "No active tab to clear.")

    def clear_grid(self):
        """Clear the entire grid."""
        while self.notebook.tabs():
            self.notebook.forget(0)  # Remove the first tab repeatedly until all are gone
        self.grid_data.clear()  # Clear stored grid data


    def create_cell(self, z_index, row, col):
        """Create a single cell in the grid."""
        data = self.grid_data[z_index]
        cell_id = f"Z{z_index}:X{col}:Y{row}"
        entry = ttk.Entry(data["grid"], width=10)
        entry.grid(row=row, column=col, padx=2, pady=2)
        entry.bind("<Button-1>", lambda event: self.select_cell(cell_id, entry))
        data["cells"][(row, col)] = {"entry": entry, "tensor": None}


    def add_cell_details_tab(self):
        """Add the Cell Details tab to the extra_notebook."""
        cell_details_tab = tk.Frame(self.extra_notebook)
        self.extra_notebook.add(cell_details_tab, text="Cell Details")

        tk.Label(cell_details_tab, text="Cell Details", font=("Arial", 16)).pack(pady=10)
        
        self.cell_details_text = tk.Text(cell_details_tab, height=15, width=60, state="disabled")
        self.cell_details_text.pack(padx=10, pady=10)
        
        # Tensor input section
        self.tensor_input_label = tk.Label(cell_details_tab, text="Add Tensor/Matrix", font=("Arial", 14))
        self.tensor_input_label.pack(pady=5)
        self.tensor_dim_label = tk.Label(cell_details_tab, text="Dimensions (e.g., 2,2):")
        self.tensor_dim_label.pack()
        self.tensor_dim_entry = tk.Entry(cell_details_tab, width=20)
        self.tensor_dim_entry.pack()
        self.tensor_values_label = tk.Label(cell_details_tab, text="Values (comma-separated):")
        self.tensor_values_label.pack()
        self.tensor_values_entry = tk.Entry(cell_details_tab, width=40)
        self.tensor_values_entry.pack()
        self.add_tensor_button = tk.Button(
            cell_details_tab, text="Add Tensor", command=self.add_tensor_to_cell
        )
        self.add_tensor_button.pack(pady=5)
        
        # Buttons for adding rows and columns
        self.add_x_button = tk.Button(cell_details_tab, text="Add X (Column)", command=self.add_x_axis)
        self.add_x_button.pack(pady=5)

        self.add_y_button = tk.Button(cell_details_tab, text="Add Y (Row)", command=self.add_y_axis)
        self.add_y_button.pack(pady=5)

        # Sidebar dropdown menu for tensor operations
        self.operation_label = tk.Label(cell_details_tab, text="Select Operation", font=("Arial", 14))
        self.operation_label.pack(pady=5)
        self.operation_var = tk.StringVar(value="Add")
        self.operation_menu = ttk.Combobox(
            cell_details_tab, textvariable=self.operation_var, state="readonly",
            values=["Add", "Subtract", "Multiply", "Dot Product", "Scalar Add"]
        )
        self.operation_menu.pack()
        self.operation_label.pack_forget()
        self.operation_menu.pack_forget()


        # buttons for clearing cells
        self.clear_selected_button = tk.Button(
            cell_details_tab, text="Clear Selected Cell", command=self.clear_selected_cell
        )
        self.clear_selected_button.pack(pady=5)

        self.clear_all_button = tk.Button(
            cell_details_tab, text="Clear All Cells", command=self.clear_all_cells
        )
        self.clear_all_button.pack(pady=5)

  
    def select_cell(self, cell_id, entry):
        """Select a cell and display its details in the Cell Details tab."""
        self.selected_cell = {"id": cell_id, "entry": entry}
        parts = cell_id.split(":")
        if len(parts) == 3:
            z_index = int(parts[0][1:])
            col = int(parts[1][1:])
            row = int(parts[2][1:])
        else:
            print(f"Invalid cell_id format: {cell_id}")
            return

        cell = self.grid_data[z_index]["cells"].get((row, col), {})
        tensor = cell.get("tensor")
        operation = cell.get("operation", "N/A")

        self.cell_details_text.configure(state="normal")
        self.cell_details_text.delete("1.0", tk.END)
        self.cell_details_text.insert("1.0", f"Cell: {cell_id}\n")
        self.cell_details_text.insert("end", f"Value: {entry.get()}\n")
        self.cell_details_text.insert("end", f"Operation: {operation}\n")
        if tensor is not None:
            self.cell_details_text.insert("end", f"Tensor:\n{tensor}\n")
        self.cell_details_text.configure(state="disabled")


    def add_tensor_to_cell(self):
        """Add a tensor/matrix to the selected cell."""
        if not self.selected_cell:
            return

        dim = self.tensor_dim_entry.get()
        values = self.tensor_values_entry.get()

        try:
            dim = tuple(map(int, dim.split(",")))
            values = list(map(float, values.split(",")))
            tensor = torch.tensor(values).view(dim)

            cell_data = self.grid_data[int(self.selected_cell["id"].split(":")[0][1:])]["cells"]
            row, col = int(self.selected_cell["id"].split(":")[2][1:]), int(self.selected_cell["id"].split(":")[1][1:])
            cell_data[(row, col)]["tensor"] = tensor
            self.selected_cell["entry"].delete(0, tk.END)
            self.selected_cell["entry"].insert(0, "Tensor")
            self.select_cell(self.selected_cell["id"], self.selected_cell["entry"])
        except Exception as e:
            print(f"Error adding tensor: {e}")

    def save_grid(self):
        """Save grid data to a JSON file."""
        file_name = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if not file_name:
            return

        grid_data = {}
        for z, data in self.grid_data.items():
            grid_data[z] = {
                "cells": {
                    f"Z{z}:X{col}:Y{row}": {
                        "value": cell["entry"].get(),
                        "tensor": cell["tensor"].tolist() if cell["tensor"] is not None else None,
                    }

                    for (row, col), cell in data["cells"].items()
                }
            }
        with open(file_name, 'w') as file:
            json.dump(grid_data, file, indent=4)
        print(f"Grid saved to {file_name}.")

    def load_grid(self):
        """Load grid data from a JSON file."""
        file_name = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not file_name:
            return

        self.clear_grid()  # Clear the current grid before loading

        with open(file_name, 'r') as file:
            loaded_data = json.load(file)

        # Load new grid data
        for z, data in loaded_data.items():
            self.add_tab()
            for key, cell_data in data["cells"].items():
                parts = key.split(":")
                if len(parts) == 3 and parts[1].startswith("X") and parts[2].startswith("Y"):
                    row = int(parts[2][1:])
                    col = int(parts[1][1:])
                    cell = self.grid_data[int(z)]["cells"][(row, col)]
                    cell["entry"].delete(0, tk.END)
                    cell["entry"].insert(0, cell_data["value"])
                    if cell_data["tensor"] is not None:
                        cell["tensor"] = torch.tensor(cell_data["tensor"])

    def perform_tensor_operations(self):
        """Perform tensor operations between layers."""
        for z in range(2, len(self.grid_data) + 1, 2):  # Process even tabs
            input_layer = self.grid_data[z - 1]["cells"]  # Odd tab before
            operation_layer = self.grid_data[z]["cells"]
            operation = self.operation_var.get()  # Use the sidebar selector
            if z + 1 not in self.grid_data:  # Ensure output tab exists
                self.add_tab()

            result_layer = self.grid_data[z + 1]["cells"]  # Odd tab after

            for (row, col), cell in input_layer.items():
                operation_cell = operation_layer.get((row,col))
                if operation_cell is None:
                        result_tensor = cell.get("tensor")  # If operation is invalid, propagate cell
    
                        # Update the result layer
                        if (row, col) not in result_layer:
                            result_layer[(row, col)] = {
                                "entry": None,
                                "tensor": result_tensor,
                                "operation": f"Z{z-1} {operation} Z{z}"
                            }

                        # Create or update the entry widget
                        if result_layer[(row, col)]["entry"] is None:
                            entry = ttk.Entry(self.grid_data[z + 1]["grid"], width=10)
                            entry.grid(row=row, column=col, padx=2, pady=2)
                            result_layer[(row, col)]["entry"] = entry

                            # Update the result layer
                        result_layer[(row, col)] = {
                                "entry": result_layer[row, col]["entry"],
                                "tensor": result_tensor,
                                "operation": f"Z{z-1} {operation} Z{z}"
                            }
                            # Display result
                        result_layer[(row, col)]["entry"].delete(0, tk.END)
                        result_layer[(row, col)]["entry"].insert(0, "Result")

                        continue
                    
            for (row, col), cell in operation_layer.items():
                input_cell = input_layer.get((row, col))
                if input_cell is None:
                        result_tensor = cell.get("tensor")  # If operation is invalid, propagate cell

                        # Update the result layer
                        if (row, col) not in result_layer:
                            result_layer[(row, col)] = {
                                "entry": None,
                                "tensor": result_tensor,
                                "operation": f"Z{z-1} {operation} Z{z}"
                            }

                        # Create or update the entry widget
                        if result_layer[(row, col)]["entry"] is None:
                            entry = ttk.Entry(self.grid_data[z + 1]["grid"], width=10)
                            entry.grid(row=row, column=col, padx=2, pady=2)
                            result_layer[(row, col)]["entry"] = entry

                            # Update the result layer
                        result_layer[(row, col)] = {
                                "entry": result_layer[row, col]["entry"],
                                "tensor": result_tensor,
                                "operation": f"Z{z-1} {operation} Z{z}"
                            }
                            # Display result
                        result_layer[(row, col)]["entry"].delete(0, tk.END)
                        result_layer[(row, col)]["entry"].insert(0, "Result")
                        continue
                else:
                    tensor_a = input_cell.get("tensor")
                    tensor_b = cell.get("tensor")

                    if tensor_a is None and tensor_b is None:
                        try:
                            tensor_a = torch.tensor([float(input_cell["entry"].get())])  # Convert scalar to tensor
                        except ValueError:                            
                            continue

                    # Handle scalar operations with valid scalar input
                    if tensor_a is not None and tensor_b is None:
                        try:
                            tensor_b = torch.tensor([float(cell["entry"].get())])  # Convert scalar to tensor
                        except ValueError:
                            # If scalar is invalid, propagate input cell unchanged
                            result_tensor=tensor_a
                        # Update the result layer
                        if (row, col) not in result_layer:
                            result_layer[(row, col)] = {
                                "entry": None,
                                "tensor": result_tensor,
                                "operation": f"Z{z-1} {operation} Z{z}"
                            }

                        # Create or update the entry widget
                        if result_layer[(row, col)]["entry"] is None:
                            entry = ttk.Entry(self.grid_data[z + 1]["grid"], width=10)
                            entry.grid(row=row, column=col, padx=2, pady=2)
                            result_layer[(row, col)]["entry"] = entry

                            # Update the result layer
                        result_layer[(row, col)] = {
                                "entry": result_layer[row, col]["entry"],
                                "tensor": result_tensor,
                                "operation": f"Z{z-1} {operation} Z{z}"
                            }
                            # Display result
                        result_layer[(row, col)]["entry"].delete(0, tk.END)
                        result_layer[(row, col)]["entry"].insert(0, "Result")
                        
                    # Handle scalar operations with valid scalar input
                    if tensor_a is None and tensor_b is not None:
                        try:
                            tensor_a = torch.tensor([float(input_cell["entry"].get())])  # Convert scalar to tensor
                        except ValueError:
                            # If scalar is invalid, propagate input cell unchanged
                            result_tensor=tensor_b
                        # Update the result layer
                        if (row, col) not in result_layer:
                            result_layer[(row, col)] = {
                                "entry": None,
                                "tensor": result_tensor,
                                "operation": f"Z{z-1} {operation} Z{z}"
                            }

                        # Create or update the entry widget
                        if result_layer[(row, col)]["entry"] is None:
                            entry = ttk.Entry(self.grid_data[z + 1]["grid"], width=10)
                            entry.grid(row=row, column=col, padx=2, pady=2)
                            result_layer[(row, col)]["entry"] = entry

                            # Update the result layer
                        result_layer[(row, col)] = {
                                "entry": result_layer[row, col]["entry"],
                                "tensor": result_tensor,
                                "operation": f"Z{z-1} {operation} Z{z}"
                            }
                            # Display result
                        result_layer[(row, col)]["entry"].delete(0, tk.END)
                        result_layer[(row, col)]["entry"].insert(0, "Result")
                        
                    if tensor_a is not None and tensor_b is not None:
                        if operation == "Add":
                            result_tensor = tensor_a + tensor_b
                        elif operation == "Subtract":
                            result_tensor = tensor_a - tensor_b
                        elif operation == "Multiply":
                            result_tensor = tensor_a * tensor_b
                        elif operation == "Dot Product":
                            result_tensor = torch.matmul(tensor_a, tensor_b)
                        elif operation == "Scalar Add":
                            result_tensor = tensor_a + tensor_b  # Scalar addition
                        else:
                            result_tensor = tensor_a  # If operation is invalid, propagate input cell

                        # Update the result layer
                        if (row, col) not in result_layer:
                            result_layer[(row, col)] = {
                                "entry": None,
                                "tensor": result_tensor,
                                "operation": f"Z{z-1} {operation} Z{z}"
                            }

                        # Create or update the entry widget
                        if result_layer[(row, col)]["entry"] is None:
                            entry = ttk.Entry(self.grid_data[z + 1]["grid"], width=10)
                            entry.grid(row=row, column=col, padx=2, pady=2)
                            result_layer[(row, col)]["entry"] = entry

                            # Update the result layer
                        result_layer[(row, col)] = {
                                "entry": result_layer[row, col]["entry"],
                                "tensor": result_tensor,
                                "operation": f"Z{z-1} {operation} Z{z}"
                            }
                            # Display result
                        result_layer[(row, col)]["entry"].delete(0, tk.END)
                        result_layer[(row, col)]["entry"].insert(0, "Result")

    def add_training_and_model_tabs(self):
        """Add Training Preparation and Model Preparation tabs."""
        training_prep_tab = tk.Frame(self.extra_notebook)
        model_prep_tab = tk.Frame(self.extra_notebook)

        self.extra_notebook.add(training_prep_tab, text="Training Preparation")
        self.extra_notebook.add(model_prep_tab, text="Model Preparation")

        self.initialize_training_prep_tab(training_prep_tab)
        self.initialize_model_prep_tab(model_prep_tab)

    def select_log_file(self):
        self.log_file_path = filedialog.asksaveasfilename(
            title="Select Log File Location",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("All files", "*.*")]
        )
        if self.log_file_path:
            print(f"Log file will be saved to: {self.log_file_path}")
        else:
            self.log_file_path = 'training_debug.log'  # Default log file
            print(f"No log file selected. Using default: {self.log_file_path}")
            
    def initialize_training_prep_tab(self, tab):
        """Initialize the Training Preparation tab."""
        tk.Label(tab, text="Training Preparation", font=("Arial", 16)).pack(pady=10)
        tk.Button(tab, text="Select Dataset Directory", command=self.select_dataset_directory).pack(pady=5)
        tk.Button(tab, text="Load Dataset", command=self.load_dataset).pack(pady=5)
        tk.Button(tab, text="Select/Create Tokenized Data", command=self.select_or_create_tokenized_data).pack(pady=5)
        tk.Button(tab, text="Tokenize Data", command=self.tokenize_data).pack(pady=5)
        tk.Button(tab, text="Start Training", command=self.start_training).pack(pady=5)
        tk.Button(tab, text="Save Model", command=self.save_model).pack(pady=5)
        tk.Button(tab, text="Stop Training", command=self.stop_training).pack(pady=5)
        self.training_mode = tk.StringVar(value="response")  # Default
        training_modes = ["imitation", "completion", "response"]
        ttk.Combobox(tab, textvariable=self.training_mode, values=training_modes, state="readonly").pack(pady=5)

        # Progress Bar
        self.progress_bar = ttk.Progressbar(tab, orient='horizontal', length=400, mode='determinate')
        self.progress_bar.pack(pady=10)
        self.status_label = ttk.Label(tab, text="Status: Ready")
        self.status_label.pack(pady=5)


    def on_device_change(self, event):
        selected_device = self.device_option.get()
        if selected_device == "GPU" and not torch.cuda.is_available():
            messagebox.showerror("Error", "GPU selected but CUDA is not available on this system.")
            self.device_option.set("CPU")
            selected_device = "CPU"
        device_str = self.map_device(selected_device)
        self.device = torch.device(device_str)
        logging.info(f"Device changed to: {self.device}")
        messagebox.showinfo("Device Selection", f"Computation device set to: {selected_device}")

    def initialize_model_prep_tab(self, tab):
        """Initialize the Model Preparation tab."""
        tk.Label(tab, text="Model Preparation", font=("Arial", 16)).pack(pady=10)
        tk.Button(tab, text="Select Tokenizer", command=self.select_tokenizer).pack(pady=5)
        tk.Button(tab, text="Load Tokenizer", command=self.load_tokenizer).pack(pady=5)
        tk.Button(tab, text="Select Model File", command=self.select_model_file).pack(pady=5)
        tk.Button(tab, text="Save Model", command=self.save_model).pack(pady=5)
        tk.Button(tab, text="Initialize/Load Model", command=self.initialize_or_load_model).pack(pady=5)


    def add_training_controls(self):
        """Add training control buttons at the bottom."""
        training_controls_frame = tk.Frame(self.root)
        training_controls_frame.pack(side="bottom", fill="x", padx=10, pady=5)

        tk.Button(training_controls_frame, text="Start Training", command=self.start_training).pack(side="left", padx=5)
        tk.Button(training_controls_frame, text="Stop Training", command=self.stop_training).pack(side="left", padx=5)
        tk.Button(training_controls_frame, text="Save Model", command=self.save_model).pack(side="left", padx=5)

    def add_numerical_parameters_tab(self):
        """Add a Numerical Parameters tab to the extra_notebook."""
        numerical_parameters_tab = tk.Frame(self.extra_notebook)
        self.extra_notebook.add(numerical_parameters_tab, text="Numerical Parameters")

        tk.Label(numerical_parameters_tab, text="Numerical Parameters", font=("Arial", 16)).pack(pady=10)

        # Parameters and their labels
        parameters = {
            "Model Name": self.model_name,
            "Vocabulary Size": self.vocab_size,
            "Hidden Size": self.hidden_size,
            "Number of Heads": self.num_heads,
            "Number of Layers": self.num_layers,
            "Batch Size": self.batch_size,
            "Learning Rate": self.learning_rate,
            "Epochs": self.epochs,
        }

        # Create a grid layout for parameters
        for row, (label_text, variable) in enumerate(parameters.items()):
            tk.Label(numerical_parameters_tab, text=label_text).pack(pady=5)
            tk.Entry(numerical_parameters_tab, textvariable=variable).pack(pady=5)

        # Add a save button
        tk.Button(
            numerical_parameters_tab,
            text="Save Parameters",
            command=self.save_numerical_parameters
        ).pack(pady=10)

    def save_numerical_parameters(self):
        """Placeholder function to save numerical parameters."""
        print(f"Model Name: {self.model_name.get()}")
        print(f"Vocabulary Size: {self.vocab_size.get()}")
        print(f"Hidden Size: {self.hidden_size.get()}")
        print(f"Number of Heads: {self.num_heads.get()}")
        print(f"Number of Layers: {self.num_layers.get()}")
        print(f"Batch Size: {self.batch_size.get()}")
        print(f"Learning Rate: {self.learning_rate.get()}")
        print(f"Epochs: {self.epochs.get()}")


    # Placeholder functions for new actions
    def select_dataset_directory(self):
        self.dataset_path = filedialog.askdirectory(title="Select Dataset Directory")
        if self.dataset_path:
            messagebox.showinfo("Success", f"Dataset directory selected: {self.dataset_path}")

    def load_dataset(self):
            """Load and preprocess dataset"""
            # Load standard dataset
            if not self.dataset_path:
                messagebox.showerror("Error", "No dataset directory selected.")
                return

            dataset_files = os.listdir(self.dataset_path)
            self.query_target_pairs = []

            for file in dataset_files:
                file_path = os.path.join(self.dataset_path, file)
                if file.endswith('.json') or file.endswith('.jsonl'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            if file.endswith('.jsonl'):
                                for line in f:
                                    conversation = json.loads(line.strip())
                                    self.query_target_pairs.extend(self.extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target = self.query_target_pairs[i]

                            else:
                                data = json.load(f)
                                self.query_target_pairs.extend(self.extract_query_target_pairs(data)) 
                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target = self.query_target_pairs[i]
                               
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to read JSON file '{file}': {str(e)}")
                else:
                    messagebox.showwarning("Warning", f"Unsupported file format: '{file}'")

            if not self.query_target_pairs:
                messagebox.showerror("Error", "No valid query/target pairs found in the dataset.")
                return

            # Store text data for saving as a text file
            self.text_data = []
            for query, target in self.query_target_pairs:
                self.text_data.append(f"User: {query}\nAssistant: {target}")

            messagebox.showinfo("Success", f"Loaded dataset with {len(self.query_target_pairs)} query/target pairs.")
            logging.info(f"Loaded dataset with {len(self.query_target_pairs)} query/target pairs.")

    def extract_query_target_pairs(self, data):
        query_target_pairs = []
        for conversation in data:
            messages = conversation.get("messages", [])
            for i in range(len(messages) - 1):
                if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
                    query = messages[i]["content"].replace('\n', ' ').strip()
                    target = messages[i + 1]["content"].replace('\n', ' ').strip()
                    query_target_pairs.append((query, target))
        return query_target_pairs

    def select_or_create_tokenized_data(self):
        use_chunked = self.use_chunked_dataset.get()
        answer = messagebox.askyesno("Select or Create Tokenized Data", "Do you want to use existing tokenized data?")
        
        if answer:
            if use_chunked:
                # User wants to use existing chunked tokenized data, select a directory
                self.tokenized_data_path = filedialog.askdirectory(
                    title="Select Tokenized Data Directory",
                    mustexist=True
                )
                if self.tokenized_data_path:
                    messagebox.showinfo("Success", f"Tokenized data directory selected: {self.tokenized_data_path}")
            else:
                # User wants to use existing single tokenized data file, select a file
                self.tokenized_data_path = filedialog.askopenfilename(
                    title="Select Tokenized Data File",
                    filetypes=[("JSON Lines files", "*.jsonl"), ("All files", "*.*")]
                )
                if self.tokenized_data_path:
                    # Attempt to load the file to validate its content
                    try:
                        with open(self.tokenized_data_path, 'r', encoding='utf-8') as f:
                            self.input_ids, self.labels = [], []
                            for line in f:
                                record = json.loads(line)
                                self.input_ids.append(record['input_ids'])
                                self.labels.append(record['labels'])
                        messagebox.showinfo("Success", f"Tokenized data file loaded: {self.tokenized_data_path}")
                        logging.info(f"Tokenized data file loaded successfully with {len(self.input_ids)} entries.")
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to load tokenized data file: {str(e)}")
        else:
            if use_chunked:
                # User wants to create new chunked tokenized data, select a directory to save
                self.tokenized_data_path = filedialog.askdirectory(
                    title="Select Directory to Save Tokenized Data"
                )
                if self.tokenized_data_path:
                    os.makedirs(self.tokenized_data_path, exist_ok=True)  # Ensure directory is created
                    messagebox.showinfo("Success", f"Tokenized data will be saved to directory: {self.tokenized_data_path}")
            else:
                # User wants to create new single tokenized data file, select a file path
                self.tokenized_data_path = filedialog.asksaveasfilename(
                    title="Save Tokenized Data As",
                    defaultextension=".jsonl",
                    filetypes=[("JSON Lines files", "*.jsonl"), ("All files", "*.*")]
                )
                if self.tokenized_data_path:
                    messagebox.showinfo("Success", f"Tokenized data will be saved to file: {self.tokenized_data_path}")
            
    def tokenize_data(self):
        if not self.tokenizer:
            messagebox.showerror("Error", "Tokenizer not loaded.")
            return
        if not hasattr(self, 'query_target_pairs') or not self.query_target_pairs:
            messagebox.showerror("Error", "No query-target pairs loaded. Please load the dataset first.")
            return
        if not self.tokenized_data_path:
            messagebox.showerror("Error", "Tokenized data path not set. Please select or create tokenized data.")
            return

        # Select training mode
        training_mode = self.training_mode.get()  # "imitation", "completion", "response"
        self.input_ids = []  # Initialize for unchunked dataset
        self.labels = []  # Initialize for unchunked dataset
        
        try:
            use_chunked = self.use_chunked_dataset.get()
            if use_chunked:
                #create path if none
                os.makedirs(self.tokenized_data_path, exist_ok=True)
                chunk_size = 32
                num_chunks = (len(self.query_target_pairs) + chunk_size - 1) // chunk_size

                for chunk_idx in range(num_chunks):
                    chunk_pairs = self.query_target_pairs[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size]
                    chunk_file_path = os.path.join(self.tokenized_data_path, f'chunk_{chunk_idx}.jsonl')

                    with open(chunk_file_path, 'w', encoding='utf-8') as f:
                        for query, target in chunk_pairs:
                            input_ids, labels = self._generate_training_pairs(query, target, training_mode)
                            if input_ids and labels:
                                record = {'input_ids': input_ids, 'labels': labels}
                                f.write(json.dumps(record) + '\n')
                logging.info(f"Chunk {chunk_idx} tokenized and saved to {chunk_file_path}")

                messagebox.showinfo("Success", f"Data tokenized into {num_chunks} chunks and saved successfully to {self.tokenized_data_path}.")
                logging.info(f"Data tokenized into {num_chunks} chunks and saved successfully to {self.tokenized_data_path}.")
            else:
                with open(self.tokenized_data_path, 'w', encoding='utf-8') as f:
                    for query, target in self.query_target_pairs:
                        input_ids, labels = self._generate_training_pairs(query, target, training_mode)

                        if input_ids and labels:
                            self.input_ids.append(input_ids)  # Store for training
                            self.labels.append(labels)  # Store for training
                            record = {'input_ids': input_ids, 'labels': labels}


                            f.write(json.dumps(record) + '\n')
                logging.info(f"Input IDs: {len(self.input_ids)} sequences loaded.")
                logging.info(f"Labels: {len(self.labels)} sequences loaded.")
                messagebox.showinfo("Success", f"Data tokenized and saved successfully to {self.tokenized_data_path}.")
                logging.info(f"Data tokenized and saved successfully to {self.tokenized_data_path}.")
        except Exception as e:
            logging.error(f"Tokenization failed: {str(e)}")
            messagebox.showerror("Error", f"Tokenization failed: {str(e)}")

    def _generate_training_pairs(self, query, target, training_mode):
        # Tokenize query and target
        query_ids = self.tokenizer.encode(query, truncation=True, max_length=max_len)
        target_ids = self.tokenizer.encode(target, truncation=True, max_length=max_len)
        # Convert tokens to integers
        query_ids = [int(token) for token in query_ids]
        target_ids = [int(token) for token in target_ids]


        if training_mode == "imitation":
            input_ids = query_ids + [self.tokenizer.eos_token_id] 
            labels = query_ids + [self.tokenizer.eos_token_id] 
        elif training_mode == "completion":
            partial_length = len(query_ids) // 2
            partial_input = query_ids[:partial_length]
            #completion = query_ids[partial_length:] + [self.tokenizer.eos_token_id]

            input_ids = partial_input + [self.tokenizer.eos_token_id]
            # For completion, we want labels to represent the entire query, not just completion
            labels = query_ids + [self.tokenizer.eos_token_id]  
        else:  # response
            input_ids = query_ids + [self.tokenizer.eos_token_id]
            labels = target_ids + [self.tokenizer.eos_token_id]

        return input_ids, labels

    def load_tokenizer(self):
        try:
            self.tokenizer_path = filedialog.askopenfilename(
                title="Select Tokenizer File",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not self.tokenizer_path or not os.path.exists(self.tokenizer_path):
                raise FileNotFoundError("Tokenizer file not selected or does not exist.")

            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.tokenizer_path)
            logging.info(f"Tokenizer loaded from {self.tokenizer_path}")

            # Load special tokens map
            special_tokens_path = os.path.join(os.path.dirname(self.tokenizer_path), "special_tokens_map.json")
            if os.path.exists(special_tokens_path):
                with open(special_tokens_path, "r") as file:
                    special_tokens = json.load(file)

                for key, value in special_tokens.items():
                    if isinstance(value, dict):
                        special_tokens[key] = AddedToken(value["content"], lstrip=value.get("lstrip", False),
                                                         rstrip=value.get("rstrip", False))
                    elif not isinstance(value, (str, AddedToken)):
                        raise ValueError(f"Invalid token format for key {key}: {value}")

                self.tokenizer.add_special_tokens(special_tokens)
                logging.info(f"Special tokens added: {special_tokens}")

            # Load tokenizer configuration
            tokenizer_config_path = os.path.join(os.path.dirname(self.tokenizer_path), "tokenizer_config.json")
            if os.path.exists(tokenizer_config_path):
                with open(tokenizer_config_path, "r") as file:
                    tokenizer_config = json.load(file)
                    self.tokenizer.init_kwargs.update(tokenizer_config)

                    # Check and set model_max_length
                    if "model_max_length" in tokenizer_config:
                        self.tokenizer.model_max_length = tokenizer_config["model_max_length"]
                    logging.info(f"Tokenizer configuration loaded: {tokenizer_config}")

            # Explicitly set model_max_length if still unset or unreasonable
            if not hasattr(self.tokenizer, "model_max_length") or self.tokenizer.model_max_length > max_len * max_len:
                self.tokenizer.model_max_length = max_len  # Default to max_len in program

            # Check consistency
            tokenizer_vocab_size = len(self.tokenizer)
            logging.info(f"Loaded tokenizer vocabulary size: {tokenizer_vocab_size}")
            self.vocab_size.set(tokenizer_vocab_size)

            # Ensure special tokens are correctly set
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = "<PAD>"
                self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("<PAD>")
                logging.warning("Pad token was not set. Defaulting to <PAD>.")
            if not self.tokenizer.unk_token:
                self.tokenizer.unk_token = "<UNK>"
                self.tokenizer.unk_token_id = self.tokenizer.convert_tokens_to_ids("<UNK>")
                logging.warning("UNK token was not set. Defaulting to <UNK>.")
            if not self.tokenizer.bos_token:
                self.tokenizer.bos_token = "<BOS>"
                self.tokenizer.bos_token_id = self.tokenizer.convert_tokens_to_ids("<BOS>")
                logging.warning("BOS token was not set. Defaulting to <BOS>.")
            if not self.tokenizer.eos_token:
                self.tokenizer.eos_token = "<EOS>"
                self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids("<EOS>")
                logging.warning("EOS token was not set. Defaulting to <EOS>.")
            print("Special tokens map:", self.tokenizer.special_tokens_map)
            print("Pad token ID:", self.tokenizer.pad_token_id)
            print("Model max length:", self.tokenizer.model_max_length)
            

        except Exception as e:
            logging.error(f"Failed to load tokenizer: {str(e)}")
            messagebox.showerror("Error", f"Failed to load tokenizer: {str(e)}")
        
    def select_tokenizer(self):
        self.tokenizer_path = filedialog.askopenfilename(
            title="Select Tokenizer File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if self.tokenizer_path:
            messagebox.showinfo("Success", f"Tokenizer file selected: {self.tokenizer_path}")
            
    def select_model_file(self):
        self.model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model Files", "*.pth;*.json"), ("All files", "*.*")]
        )
        if self.model_path:
            if self.model_path.endswith('.json'):
                # Load model configuration
                with open(self.model_path, 'r') as f:
                    config = json.load(f)
                # Update GUI parameters
                self.vocab_size.set(config.get("vocab_size", self.vocab_size.get()))
                self.hidden_size.set(config.get("embed_size", self.hidden_size.get()))
                self.num_heads.set(config.get("num_heads", self.num_heads.get()))
                self.num_nodes.set(config.get("num_nodes", self.num_nodes.get()))

                self.num_layers.set(config.get("num_layers", self.num_layers.get()))
                self.architecture.set(config.get("architecture", self.architecture.get()))
                messagebox.showinfo("Success", f"Model configuration loaded from: {self.model_path}")
            elif self.model_path.endswith('.pth'):
                # Load model weights
                config_directory = os.path.dirname(self.model_path)
                config_path = os.path.join(config_directory, 'model_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    # Update GUI parameters
                    self.vocab_size.set(config.get("vocab_size", self.vocab_size.get()))
                    self.hidden_size.set(config.get("embed_size", self.hidden_size.get()))
                    self.num_heads.set(config.get("num_heads", self.num_heads.get()))
                    self.num_nodes.set(config.get("num_nodes", self.num_nodes.get()))

                    self.num_layers.set(config.get("num_layers", self.num_layers.get()))
                    self.architecture.set(config.get("architecture", self.architecture.get()))
                    # Load the model
                    self.load_model()
                    # Load model state
                    state_dict = torch.load(self.model_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    messagebox.showinfo("Success", f"Model weights and configuration loaded from: {self.model_path}")
                else:
                    messagebox.showwarning("Warning", "Model configuration file not found. Please ensure the configuration is set correctly.")
            else:
                messagebox.showerror("Error", "Unsupported file format selected.")

    def initialize_or_load_model(self):
        try:
            if not self.tokenizer:
                vocab_size = self.vocab_size.get()
            else:
                vocab_size = len(self.tokenizer)

            # Log and validate vocab size
            logging.info(f"Tokenizer vocabulary size: {vocab_size}")
            self.vocab_size.set(vocab_size)

            # Initialize the model based on architecture
            if self.architecture.get() == "Cell Transformer":
                self.model = Cellformer(
                    vocab_size=vocab_size,
                    embed_size=self.hidden_size.get(),
                    hidden_size=self.hidden_size.get(),
                    num_heads=self.num_heads.get(),
                    num_layers=self.num_layers.get(),
                    max_seq_length=max_len
                )
            elif self.architecture.get() == "Cell Ternary Transformer":
                self.model = Cellternaryformer(
                    vocab_size=vocab_size,
                    embed_size=self.hidden_size.get(),
                    hidden_size=self.hidden_size.get(),
                    num_heads=self.num_heads.get(),
                    max_seq_length=max_len
                )
            else:
                messagebox.showerror("Error", f"Unsupported architecture: {self.architecture.get()}")
                return

            # Move the entire model to the selected device

            self.model.to(self.device)
            logging.info(f"Model moved to device: {self.device}")

            # Resize embeddings to match tokenizer vocabulary size
            if hasattr(self.model, 'resize_token_embeddings'):
                self.model.resize_token_embeddings(vocab_size)
                logging.info("Embeddings resized to match tokenizer vocabulary size.")

            # Load checkpoint if a model file is selected
            if self.model_path and self.model_path.endswith('.pth'):
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=True)
                logging.info("Model weights loaded and resized successfully.")

            logging.info(f"Model initialized on device: {self.device}")
            messagebox.showinfo("Success", "Model initialized successfully.")

        except Exception as e:
            logging.error(f"Failed to initialize model: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize model: {str(e)}")

    def start_training(self):
        # Start training in a separate thread to keep the GUI responsive
        self.stop_training.clear()
        training_thread = threading.Thread(target=self.training_loop)
        training_thread.start()

    def update_progress(self, progress_value):
        self.progress_bar['value'] = progress_value

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")

    def save_checkpoint(self, model, optimizer, epoch, path):
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError(f"Expected path to be str or os.PathLike, got {type(path).__name__}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        


    def validate_training_parameters(self):
        # Validate batch size
        try:
            batch_size = int(self.batch_size.get())
            if batch_size <= 0:
                raise ValueError
        except (TypeError, ValueError):
            logging.error(f"Invalid batch size: {self.batch_size.get()}")
            messagebox.showerror("Error", "Batch size must be a positive integer.")
            return False

        # Validate epochs
        try:
            epochs = int(self.epochs.get())
            if epochs <= 0:
                raise ValueError
        except (TypeError, ValueError):
            logging.error(f"Invalid epochs value: {self.epochs.get()}")
            messagebox.showerror("Error", "Epochs must be a positive integer.")
            return False

        if not self.tokenized_data_path or not os.path.exists(self.tokenized_data_path):
            logging.error("Tokenized data path is invalid or does not exist.")
            messagebox.showerror("Error", "Tokenized data is not selected or does not exist.")
            return False

        if not hasattr(self.tokenizer, 'pad_token_id') or self.tokenizer.pad_token_id is None:
            logging.error("Tokenizer pad_token_id is not set.")
            messagebox.showerror("Error", "Tokenizer is missing pad_token_id.")
            return False

        return True

    def stop_training(self):
        self.stop_training.set()
        messagebox.showinfo("Stop Training", "Training stopped.")
        logging.info("Training stopped by user.")

    def save_model(self):
        if not self.model:
            messagebox.showerror("Error", "Model has not been initialized. Cannot save.")
            logging.error("Attempted to save model but model was not initialized.")
            return
        if not self.tokenizer:
            messagebox.showerror("Error", "Tokenizer has not been initialized. Cannot save.")
            logging.error("Attempted to save model but tokenizer was not initialized.")
            return

        save_directory = filedialog.askdirectory(title="Select Save Directory")
        if save_directory:
            config = {
            "vocab_size": self.vocab_size.get(),
            "embed_size": self.hidden_size.get(),
            "hidden_size": self.hidden_size.get(),
            "num_heads": self.num_heads.get(),
            "num_layers": self.num_layers.get(),
            "architecture": self.architecture.get(),
            "num_parameters": self.num_parameters.get(),
            "layers": self.layers
        }
            config_path = os.path.join(save_directory, 'model_config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f)


            # Save the model state dictionary
            if self.architecture.get() == "Cascade Transformer":
                model_file_name = 'cascade_transformer_model.pth'
            elif self.architecture.get() == "Cascade MatMul-Free LM":
                model_file_name = 'cascade_matmul_free_lm.pth'
            else:
                messagebox.showerror("Error", f"Unsupported architecture: {self.architecture.get()}")
                return

            model_path = os.path.join(save_directory, model_file_name)
            torch.save(self.model.state_dict(), model_path)

            # Save the tokenizer
            self.tokenizer.save_pretrained(save_directory)

            messagebox.showinfo("Success", "Model, tokenizer, and config saved successfully.")
            logging.info("Model, tokenizer, and config saved successfully.")


# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = TensorCellGUI(root)
    root.mainloop()
