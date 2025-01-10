import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import torch


class TensorCellGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Tensor Cell-Based GUI with Tensors")
        self.root.geometry("1400x800")

        # Notebook for tabs (Z-dimension)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        # Side panel for cell details
        self.side_panel = tk.Frame(root, width=300)
        self.side_panel.pack(side="right", fill="y", padx=10)
        self.cell_details_label = tk.Label(self.side_panel, text="Cell Details", font=("Arial", 14))
        self.cell_details_label.pack(pady=5)
        self.cell_details_text = tk.Text(self.side_panel, height=10, width=40, state="disabled")
        self.cell_details_text.pack()

        # Tensor input section
        self.tensor_input_label = tk.Label(self.side_panel, text="Add Tensor/Matrix", font=("Arial", 14))
        self.tensor_input_label.pack(pady=5)
        self.tensor_dim_label = tk.Label(self.side_panel, text="Dimensions (e.g., 2,2):")
        self.tensor_dim_label.pack()
        self.tensor_dim_entry = tk.Entry(self.side_panel, width=20)
        self.tensor_dim_entry.pack()
        self.tensor_values_label = tk.Label(self.side_panel, text="Values (comma-separated):")
        self.tensor_values_label.pack()
        self.tensor_values_entry = tk.Entry(self.side_panel, width=40)
        self.tensor_values_entry.pack()
        self.add_tensor_button = tk.Button(
            self.side_panel, text="Add Tensor", command=self.add_tensor_to_cell
        )
        self.add_tensor_button.pack(pady=5)

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

        # Sidebar dropdown menu for tensor operations
        self.operation_label = tk.Label(self.side_panel, text="Select Operation", font=("Arial", 14))
        self.operation_label.pack(pady=5)
        self.operation_var = tk.StringVar(value="Add")
        self.operation_menu = ttk.Combobox(
            self.side_panel, textvariable=self.operation_var, state="readonly",
            values=["Add", "Subtract", "Multiply", "Dot Product", "Scalar Add"]
        )
        self.operation_menu.pack()
        self.operation_label.pack_forget()
        self.operation_menu.pack_forget()


        # buttons for clearing cells
        self.clear_selected_button = tk.Button(
            self.side_panel, text="Clear Selected Cell", command=self.clear_selected_cell
        )
        self.clear_selected_button.pack(pady=5)

        self.clear_all_button = tk.Button(
            self.side_panel, text="Clear All Cells", command=self.clear_all_cells
        )
        self.clear_all_button.pack(pady=5)

        self.selected_cell = None  # Track selected cell
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

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
        """Add a new tab for a new Z-layer."""
        z_index = len(self.notebook.tabs()) + 1
        frame = tk.Frame(self.notebook)
        self.notebook.add(frame, text=f"Z={z_index}")

        # Initialize grid for this tab
        grid = ttk.Frame(frame)
        grid.pack(fill="both", expand=True)

        self.grid_data[z_index] = {
            "grid": grid,
            "cells": {},
            "rows": 5,
            "columns": 5,
            "operation": None if z_index % 2 == 1 else "Add"  # Default operation for even tabs
        }

        self.initialize_grid(z_index)


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
        """Clear all cells in all tabs."""
        confirm = messagebox.askyesno("Confirm Clear All", "Are you sure you want to clear all cells?")
        if confirm:
            for z_index, data in self.grid_data.items():
                for (row, col), cell in data["cells"].items():
                    cell["entry"].delete(0, tk.END)
                    cell["tensor"] = None
            self.cell_details_text.configure(state="normal")
            self.cell_details_text.delete("1.0", tk.END)
            self.cell_details_text.insert("1.0", "All cells cleared.\n")
            self.cell_details_text.configure(state="disabled")

    def clear_grid(self):
        """Clear the active layer."""
        active_tab_index = self.notebook.index(self.notebook.select()) + 1  # Tabs are 1-indexed
        if active_tab_index in self.grid_data:
            data = self.grid_data[active_tab_index]
            for (row, col), cell in data["cells"].items():
                cell["entry"].delete(0, tk.END)
                cell["tensor"] = None
            messagebox.showinfo("Grid Cleared", f"Layer Z={active_tab_index} has been cleared.")
        else:
            messagebox.showinfo("No Active Tab", "No active tab to clear.")

    def create_cell(self, z_index, row, col):
        """Create a single cell in the grid."""
        data = self.grid_data[z_index]
        cell_id = f"Z{z_index}:X{col}:Y{row}"
        entry = ttk.Entry(data["grid"], width=10)
        entry.grid(row=row, column=col, padx=2, pady=2)
        entry.bind("<Button-1>", lambda event: self.select_cell(cell_id, entry))
        data["cells"][(row, col)] = {"entry": entry, "tensor": None}

    def select_cell(self, cell_id, entry):
        """Select a cell and display its details in the side panel."""
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


# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = TensorCellGUI(root)
    root.mainloop()
