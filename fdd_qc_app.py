import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import os
import json
import fitz  # PyMuPDF
import sys
from PIL import Image, ImageTk
import io
import re
from pdf_processor import PDFProcessor, JSONProcessor, VerificationEngine

class FlaggedPairSelector(tk.Toplevel):
    """Dialog to select a flagged file pair."""
    def __init__(self, parent, flagged_pairs_dict):
        super().__init__(parent)
        self.title("Select Flagged Pair")
        self.geometry("400x300")
        self.transient(parent) # Stay on top of parent
        self.grab_set() # Modal behavior

        self.flagged_pairs = flagged_pairs_dict
        self.selected_id = None

        label = ttk.Label(self, text="Select a file ID to review:")
        label.pack(pady=10)

        self.listbox = tk.Listbox(self, selectmode=tk.SINGLE)
        self.listbox.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        for file_id in sorted(self.flagged_pairs.keys()):
            # Display the ID and maybe part of the filename for context
            pdf_name = os.path.basename(self.flagged_pairs[file_id]['pdf'])
            self.listbox.insert(tk.END, f"{file_id} ({pdf_name})")

        self.listbox.bind("<Double-Button-1>", self._on_load)

        button_frame = ttk.Frame(self)
        button_frame.pack(pady=10)

        load_button = ttk.Button(button_frame, text="Load", command=self._on_load)
        load_button.pack(side=tk.LEFT, padx=5)

        cancel_button = ttk.Button(button_frame, text="Cancel", command=self.destroy)
        cancel_button.pack(side=tk.LEFT, padx=5)

        self.wait_window() # Wait until the window is closed

    def _on_load(self, event=None):
        selection_index = self.listbox.curselection()
        if selection_index:
            full_text = self.listbox.get(selection_index[0])
            # Extract the ID (first part before the space)
            self.selected_id = full_text.split(" (", 1)[0]
            self.destroy() # Close the dialog

class FDDQualityControlApp:
    """
    Main application for FDD Quality Control System (Batch Review Mode)
    """
    
    def __init__(self, root, flagged_for_review: dict):
        """
        Initialize the application in batch review mode.
        
        Args:
            root: Tkinter root window
            flagged_for_review (dict): Dictionary of pairs flagged during batch processing. 
                                       Format: {file_id: {'pdf': pdf_path, 'json': json_path, 'results': results_path}}
        """
        self.root = root
        self.root.title("FDD Header QC Review")
        self.root.geometry("1400x800")
        
        # Bind Enter key to the root window
        self.root.bind("<Return>", self.on_header_enter)
        
        self.flagged_pairs = flagged_for_review
        self.corrected_files = self._load_corrected_files_list()
        self.loaded_file_id = None # Track the currently loaded flagged ID

        # Data storage for the currently loaded pair
        self.current_pdf_path = None
        self.current_json_path = None
        self.current_results_path = None
        self.pdf_processor = None
        self.json_processor = None
        self.verification_results = {} # Store loaded results here
        
        # PDF display variables
        self.current_page = 1
        self.zoom_factor = 1.0
        self.pdf_document = None
        
        # Create UI components
        self.create_menu()
        self.create_main_layout()
        
        # Initialize status
        self.update_status(f"Ready. {len(self.flagged_pairs)} pairs flagged for review.")
        
        # Auto-load the first uncorrected file
        self.root.after(100, self._auto_load_next_uncorrected)

    def _load_corrected_files_list(self) -> set:
        """Load the list of already corrected file IDs."""
        corrected_files = set()
        corrected_file_path = os.path.join(os.path.dirname(__file__), "output", "corrected_files.json")
        if os.path.exists(corrected_file_path):
            try:
                with open(corrected_file_path, 'r') as f:
                    corrected_files = set(json.load(f))
                print(f"Loaded {len(corrected_files)} previously corrected files.")
            except Exception as e:
                print(f"Error loading corrected files list: {e}")
        return corrected_files
    
    def _save_corrected_files_list(self):
        """Save the updated list of corrected file IDs."""
        corrected_file_path = os.path.join(os.path.dirname(__file__), "output", "corrected_files.json")
        try:
            with open(corrected_file_path, 'w') as f:
                json.dump(list(self.corrected_files), f)
            print(f"Saved {len(self.corrected_files)} corrected files to list.")
        except Exception as e:
            print(f"Error saving corrected files list: {e}")
    
    def _get_uncorrected_files(self) -> list:
        """Get list of flagged file IDs that haven't been corrected yet."""
        return [file_id for file_id in self.flagged_pairs.keys() 
                if file_id not in self.corrected_files]
    
    def _auto_load_next_uncorrected(self):
        """Automatically load the next uncorrected file."""
        uncorrected_files = self._get_uncorrected_files()
        if not uncorrected_files:
            messagebox.showinfo("All Files Processed", 
                               "All flagged files have been corrected. You can reopen files manually if needed.")
            return
        
        # Sort by ID to ensure consistent order
        next_file_id = sorted(uncorrected_files)[0]
        self._load_flagged_pair(next_file_id)
        
        # Update status to indicate auto-loading
        remaining = len(uncorrected_files)
        self.update_status(f"Auto-loaded file {next_file_id}. {remaining} uncorrected file(s) remaining.")
    
    def create_menu(self):
        """Create the application menu"""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Flagged Pair...", command=self._show_flagged_pairs_dialog)
        file_menu.add_command(label="Load Next Uncorrected", command=self._auto_load_next_uncorrected)
        file_menu.add_separator()
        file_menu.add_command(label="Save Corrected JSON", command=self.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Help", command=self.show_help_batch)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)

    def create_main_layout(self):
        """Create the main application layout"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a PanedWindow for resizable sections
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - PDF viewer
        self.pdf_frame = ttk.LabelFrame(paned_window, text="PDF Viewer")
        paned_window.add(self.pdf_frame, weight=2)
        
        # PDF canvas with scrollbars
        pdf_canvas_frame = ttk.Frame(self.pdf_frame)
        pdf_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add vertical scrollbar
        self.pdf_vscrollbar = ttk.Scrollbar(pdf_canvas_frame, orient=tk.VERTICAL)
        self.pdf_vscrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add horizontal scrollbar
        self.pdf_hscrollbar = ttk.Scrollbar(pdf_canvas_frame, orient=tk.HORIZONTAL)
        self.pdf_hscrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # PDF canvas
        self.pdf_canvas = tk.Canvas(
            pdf_canvas_frame, 
            bg="white",
            xscrollcommand=self.pdf_hscrollbar.set,
            yscrollcommand=self.pdf_vscrollbar.set
        )
        self.pdf_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Configure scrollbars
        self.pdf_vscrollbar.config(command=self.pdf_canvas.yview)
        self.pdf_hscrollbar.config(command=self.pdf_canvas.xview)
        
        # Bind mouse wheel events
        self.pdf_canvas.bind("<MouseWheel>", self._on_mousewheel)  # Windows
        self.pdf_canvas.bind("<Button-4>", self._on_mousewheel)    # Linux scroll up
        self.pdf_canvas.bind("<Button-5>", self._on_mousewheel)    # Linux scroll down
        
        # PDF navigation frame
        pdf_nav_frame = ttk.Frame(self.pdf_frame)
        pdf_nav_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(pdf_nav_frame, text="Previous", command=self.prev_page).pack(side=tk.LEFT, padx=5)
        ttk.Button(pdf_nav_frame, text="Next", command=self.next_page).pack(side=tk.LEFT, padx=5)
        
        self.page_var = tk.StringVar(value="Page: 0 / 0")
        ttk.Label(pdf_nav_frame, textvariable=self.page_var).pack(side=tk.LEFT, padx=20)
        
        ttk.Button(pdf_nav_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.RIGHT, padx=5)
        ttk.Button(pdf_nav_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.RIGHT, padx=5)
        
        # Right panel - Headers and verification
        self.headers_frame = ttk.LabelFrame(paned_window, text="FDD Headers for Review")
        paned_window.add(self.headers_frame, weight=1)
        
        # Create headers table
        self.create_headers_table()
        
        # Bottom panel - Status and controls
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(bottom_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=5)
        
        # Action buttons
        ttk.Button(bottom_frame, text="Save Corrected JSON", command=self.save_results).pack(side=tk.RIGHT, padx=5)
    
    def create_headers_table(self):
        """Create the headers table for displaying verification results"""
        # Create a frame for the table
        table_frame = ttk.Frame(self.headers_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create scrollbars
        y_scrollbar = ttk.Scrollbar(table_frame)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create Treeview widget
        columns = ("item", "header", "expected_page", "found_page", "status")
        self.headers_table = ttk.Treeview(table_frame, columns=columns, show="headings", yscrollcommand=y_scrollbar.set)
        
        # Configure scrollbars
        y_scrollbar.config(command=self.headers_table.yview)
        
        # Define column headings
        self.headers_table.heading("item", text="Item #")
        self.headers_table.heading("header", text="Header Text")
        self.headers_table.heading("expected_page", text="Expected Page")
        self.headers_table.heading("found_page", text="Found Page")
        self.headers_table.heading("status", text="Status")
        
        # Define column widths
        self.headers_table.column("item", width=50, anchor=tk.CENTER)
        self.headers_table.column("header", width=250)
        self.headers_table.column("expected_page", width=100, anchor=tk.CENTER)
        self.headers_table.column("found_page", width=100, anchor=tk.CENTER)
        self.headers_table.column("status", width=100, anchor=tk.CENTER)
        
        # Pack the table
        self.headers_table.pack(fill=tk.BOTH, expand=True)
        
        # Bind events
        self.headers_table.bind("<Double-1>", self.on_header_double_click)
        self.headers_table.bind("<<TreeviewSelect>>", self.on_header_select)
        self.headers_table.bind("<Return>", self.on_header_enter)  # Bind Enter key to approve action
        
        # Create detail frame below the table
        self.detail_frame = ttk.LabelFrame(self.headers_frame, text="Header Details")
        self.detail_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create fields for editing
        edit_frame = ttk.Frame(self.detail_frame)
        edit_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(edit_frame, text="Item Number:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.item_number_var = tk.StringVar()
        ttk.Label(edit_frame, textvariable=self.item_number_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(edit_frame, text="Header Text:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.header_text_var = tk.StringVar()
        ttk.Label(edit_frame, textvariable=self.header_text_var, wraplength=300).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(edit_frame, text="Expected Page:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.expected_page_var = tk.StringVar()
        ttk.Entry(edit_frame, textvariable=self.expected_page_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(edit_frame, text="Found Page:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.found_page_var = tk.StringVar()
        ttk.Label(edit_frame, textvariable=self.found_page_var).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(edit_frame, text="Confidence:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.confidence_var = tk.StringVar()
        ttk.Label(edit_frame, textvariable=self.confidence_var).grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Action buttons
        button_frame = ttk.Frame(self.detail_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Go to Expected Page", command=self.go_to_expected_page).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Go to Found Page", command=self.go_to_found_page).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Update Page Number", command=self.update_page_number).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Approve", command=self.approve_header).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Reject", command=self.reject_header).pack(side=tk.RIGHT, padx=5)
    
    def _show_flagged_pairs_dialog(self):
        if not self.flagged_pairs:
             messagebox.showinfo("Information", "No pairs were flagged for review.")
             return
        
        dialog = FlaggedPairSelector(self.root, self.flagged_pairs)
        selected_id = dialog.selected_id

        if selected_id:
            self._load_flagged_pair(selected_id)
            
    def _load_flagged_pair(self, file_id: str):
        """Loads the PDF, original JSON, and results JSON for the selected flagged pair."""
        if file_id not in self.flagged_pairs:
            messagebox.showerror("Error", f"File ID {file_id} not found in flagged list.")
            return

        pair_info = self.flagged_pairs[file_id]
        pdf_path = pair_info['pdf']
        json_path = pair_info['json']
        results_path = pair_info['results']

        self.update_status(f"Loading flagged pair ID: {file_id}...")
        try:
            # --- Load PDF ---
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            self.pdf_document = fitz.open(pdf_path)
            self.current_pdf_path = pdf_path
            self.pdf_processor = PDFProcessor(pdf_path) # Still useful for page count, text access? Maybe optional.
            self.current_page = 1
            
            # --- Load Original JSON ---
            if not os.path.exists(json_path):
                 raise FileNotFoundError(f"Original JSON not found: {json_path}")
            self.json_processor = JSONProcessor(json_path) # Loads original headers
            self.current_json_path = json_path
            
            # --- Load Verification Results ---
            if not os.path.exists(results_path):
                 raise FileNotFoundError(f"Results JSON not found: {results_path}")
            with open(results_path, 'r') as f_res:
                # Load results, converting string keys back to int if needed (JSON saves keys as strings)
                loaded_results = json.load(f_res)
                self.verification_results = {int(k): v for k, v in loaded_results.items()}

            self.loaded_file_id = file_id

            # --- Update UI ---
            self.populate_headers_table() # Populate with original JSON data first
            self.update_headers_table()   # Update with loaded verification results
            self.update_page_display()    # Show PDF page 1
            
            self.update_status(f"Loaded flagged pair: {file_id}. PDF: {os.path.basename(pdf_path)}")

        except Exception as e:
            messagebox.showerror("Error Loading Pair", f"Failed to load pair ID {file_id}:\n{str(e)}")
            # Reset state maybe?
            self.pdf_document = None
            self.pdf_processor = None
            self.json_processor = None
            self.verification_results = {}
            self.loaded_file_id = None
            self.update_status("Error loading pair.")
            self.clear_headers_table() # Clear table on error
            self.pdf_canvas.delete("all") # Clear PDF view
            self.page_var.set("Page: 0 / 0")

    def populate_headers_table(self):
        """Populate the headers table with data from the loaded original JSON file"""
        self.clear_headers_table() # Clear first
        
        # Add headers from the loaded JSONProcessor
        if self.json_processor:
            headers = self.json_processor.get_all_headers()
            for header in headers:
                item_number = header.get('item_number')
                header_text = header.get('text')
                page_number = header.get('page_number') # Original page number
                self.headers_table.insert("", "end", values=(
                    item_number,
                    header_text,
                    page_number,
                    "",  # Found page (will be filled by update_headers_table)
                    "Loading..."  # Initial status (will be updated)
                ))

    def clear_headers_table(self):
         """Clears all items from the headers table."""
         for item in self.headers_table.get_children():
            self.headers_table.delete(item)

    def update_headers_table(self):
        """Update the headers table with loaded verification results"""
        if not self.verification_results:
            print("No verification results loaded to update table.")
            # Maybe set all statuses to 'N/A' or 'Error'?
            for item in self.headers_table.get_children():
                 values = list(self.headers_table.item(item, "values"))
                 values[4] = "Results N/A" # Update status column
                 self.headers_table.item(item, values=tuple(values))
            return
        
        # Update each row with verification results
        items_in_table = {int(self.headers_table.item(item, "values")[0]): item for item in self.headers_table.get_children()}

        for item_number, result in self.verification_results.items():
            item_id = items_in_table.get(item_number)
            if not item_id:
                print(f"Warning: Result found for item {item_number}, but not in table.")
                continue

            # Update values in the existing row
            self.headers_table.item(item_id, values=(
                item_number,
                result.get('header_text', 'N/A'), # Use result data if available
                result.get('expected_page', 'N/A'),
                result.get('best_match_page', "Not found") if result.get('best_match_page') is not None else "Not found",
                result.get('status', 'unknown').replace("_", " ").title()
            ))
            
            # Set row color based on status
            status = result.get('status', 'unknown')
            tag = status # Use status directly as tag
            if status == "verified":
                 self.headers_table.item(item_id, tags=(tag,))
            elif status == "likely_correct":
                 self.headers_table.item(item_id, tags=(tag,))
            elif status == "needs_review":
                 self.headers_table.item(item_id, tags=(tag,))
            elif status == "likely_incorrect":
                 self.headers_table.item(item_id, tags=(tag,))
            elif status == "not_found":
                 self.headers_table.item(item_id, tags=(tag,))
            else:
                 self.headers_table.item(item_id, tags=("unknown",)) # Fallback tag

        # Configure tag colors (ensure all potential status tags are configured)
        self.headers_table.tag_configure("verified", background="#c8e6c9")
        self.headers_table.tag_configure("likely_correct", background="#dcedc8")
        self.headers_table.tag_configure("needs_review", background="#fff9c4")
        self.headers_table.tag_configure("likely_incorrect", background="#ffccbc")
        self.headers_table.tag_configure("not_found", background="#cfd8dc")
        self.headers_table.tag_configure("unknown", background="#ffffff") # Default white

    def on_header_select(self, event):
        """Handle header selection in the table"""
        selection = self.headers_table.selection()
        if not selection:
            return
        
        # Get selected item
        item = selection[0]
        values = self.headers_table.item(item, "values")
        
        # Update detail view
        try:
            item_number = int(values[0])
            self.item_number_var.set(item_number)
            self.header_text_var.set(values[1])
            self.expected_page_var.set(values[2]) # This is the *original* expected page from JSON/Results
            self.found_page_var.set(values[3]) # This is the page found by verification
            
            # Get confidence from verification results
            # Ensure verification_results uses integer keys if loaded from JSON
            if self.verification_results and item_number in self.verification_results:
                result = self.verification_results[item_number]
                confidence = result.get('confidence', 0)
                self.confidence_var.set(f"{confidence:.2f}")
            else:
                self.confidence_var.set("N/A")
                
            # Automatically navigate to the expected page when a header is selected
            self.go_to_expected_page()
            
        except (ValueError, IndexError):
             # Handle cases where table might be empty or have unexpected values
             self.item_number_var.set("")
             self.header_text_var.set("")
             self.expected_page_var.set("")
             self.found_page_var.set("")
             self.confidence_var.set("")
             print("Error updating detail view from table selection.")

    def on_header_double_click(self, event):
        """Handle double-click on a header in the table"""
        self.go_to_expected_page()
    
    def go_to_expected_page(self):
        """Go to the expected page in the PDF (from original JSON/results)."""
        if not self.pdf_document: return
        try:
            expected_page_str = self.expected_page_var.get()
            if not expected_page_str or expected_page_str == "N/A":
                 messagebox.showwarning("Warning", "No expected page number available.")
                 return
            expected_page = int(expected_page_str)
            if expected_page > 0 and expected_page <= self.pdf_document.page_count:
                self.current_page = expected_page
                self.update_page_display()
            else:
                messagebox.showwarning("Warning", f"Invalid page number: {expected_page}")
        except ValueError:
            messagebox.showwarning("Warning", "Invalid page number format.")
        except AttributeError: # Handle case where pdf_document might be None
             messagebox.showerror("Error", "PDF document not loaded.")

    def go_to_found_page(self):
        """Go to the found page in the PDF"""
        found_page = self.found_page_var.get()
        if found_page and found_page != "Not found":
            try:
                page_num = int(found_page)
                if page_num > 0 and page_num <= self.pdf_document.page_count:
                    self.current_page = page_num
                    self.update_page_display()
                else:
                    messagebox.showwarning("Warning", "Invalid page number.")
            except ValueError:
                messagebox.showwarning("Warning", "Invalid page number.")
        else:
            messagebox.showinfo("Information", "No matching page found.")
    
    def update_page_number(self):
        """Update the page number for the selected header"""
        selection = self.headers_table.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a header first.")
            return
        
        try:
            # Get the new page number
            new_page = int(self.expected_page_var.get())
            
            # Get the item number
            item = selection[0]
            values = self.headers_table.item(item, "values")
            item_number = int(values[0])
            
            # Update in JSON processor
            if self.json_processor:
                self.json_processor.update_header_page_number(item_number, new_page)
            
            # Update in verification results if available
            if self.verification_results and item_number in self.verification_results:
                self.verification_results[item_number]['expected_page'] = new_page
            
            # Update table
            self.headers_table.item(item, values=(
                item_number,
                values[1],
                new_page,
                values[3],
                "Updated"
            ))
            
            self.update_status(f"Updated page number for Item {item_number} to {new_page}.")
        
        except ValueError:
            messagebox.showwarning("Warning", "Invalid page number.")
    
    def approve_header(self):
        """Approve the selected header"""
        selection = self.headers_table.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a header first.")
            return
        
        item = selection[0]
        values = self.headers_table.item(item, "values")
        try:
             item_number = int(values[0])
        except (ValueError, IndexError):
             print("Could not get item number from selected row.")
             return

        # Update in verification results (which holds the state during review)
        if item_number in self.verification_results:
            self.verification_results[item_number]['status'] = "verified"
            # Update the expected page in results if it was manually changed in the entry
            try:
                current_entry_page = int(self.expected_page_var.get())
                self.verification_results[item_number]['expected_page'] = current_entry_page
                # Also update the underlying json_processor if loaded
                if self.json_processor:
                     self.json_processor.update_header_page_number(item_number, current_entry_page)
            except ValueError:
                 print(f"Could not update expected page for item {item_number} from entry.")

        # Update table visually
        # Use data from verification_results to ensure consistency
        result = self.verification_results.get(item_number, {})
        self.headers_table.item(item, values=(
            item_number,
            result.get('header_text', values[1]),
            result.get('expected_page', values[2]),
            result.get('best_match_page', values[3]) if result.get('best_match_page') is not None else "Not found",
            "Verified"
        ))
        self.headers_table.item(item, tags=("verified",))
        
        self.update_status(f"Approved Item {item_number}.")

    def reject_header(self):
        """Reject the selected header"""
        selection = self.headers_table.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a header first.")
            return

        item = selection[0]
        values = self.headers_table.item(item, "values")
        try:
             item_number = int(values[0])
        except (ValueError, IndexError):
             print("Could not get item number from selected row.")
             return

        # Update in verification results (which holds the state during review)
        if item_number in self.verification_results:
            self.verification_results[item_number]['status'] = "needs_review" # Mark as needs review again
            # Update the expected page in results if it was manually changed in the entry
            try:
                 current_entry_page = int(self.expected_page_var.get())
                 self.verification_results[item_number]['expected_page'] = current_entry_page
                 # Also update the underlying json_processor if loaded
                 if self.json_processor:
                      self.json_processor.update_header_page_number(item_number, current_entry_page)
            except ValueError:
                 print(f"Could not update expected page for item {item_number} from entry.")

        # Update table visually
        result = self.verification_results.get(item_number, {})
        self.headers_table.item(item, values=(
            item_number,
            result.get('header_text', values[1]),
            result.get('expected_page', values[2]),
            result.get('best_match_page', values[3]) if result.get('best_match_page') is not None else "Not found",
            "Needs Review"
        ))
        self.headers_table.item(item, tags=("needs_review",))

        self.update_status(f"Rejected Item {item_number}. Marked for review.")

    def save_results(self):
        """Save the currently loaded, potentially corrected, header data."""
        if not self.json_processor or not self.current_json_path:
            messagebox.showwarning("Warning", "No JSON data loaded to save.")
            return
        
        # Create the output directory if it doesn't exist
        corrected_output_dir = os.path.join(os.path.dirname(__file__), "output", "corrected_json")
        os.makedirs(corrected_output_dir, exist_ok=True)
        
        # Generate filename based on the original JSON, adding '_corrected'
        original_basename = os.path.basename(self.current_json_path)
        suggested_filename = original_basename.replace(".json", "_corrected.json")
        if suggested_filename == original_basename: # Ensure suffix is added
            suggested_filename = os.path.splitext(original_basename)[0] + "_corrected.json"
        
        # Full path to save the file
        output_path = os.path.join(corrected_output_dir, suggested_filename)
        
        try:
            # Save the data currently held by json_processor (which includes edits)
            saved_path = self.json_processor.save_json(output_path)
            messagebox.showinfo("Success", f"Corrected headers saved to {os.path.basename(saved_path)}")
            self.update_status(f"Saved corrected data to {os.path.basename(saved_path)}")
            
            # Mark current file as corrected
            if self.loaded_file_id:
                self.corrected_files.add(self.loaded_file_id)
                self._save_corrected_files_list()
            
            # Auto-load next file after a short delay
            self.root.after(500, self._auto_load_next_uncorrected)
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save corrected data: {str(e)}")
            self.update_status("Error saving corrected data")

    def update_page_display(self):
        """Update the PDF page display"""
        if not self.pdf_document:
            return
        
        # Clear canvas
        self.pdf_canvas.delete("all")
        
        # Get the page
        page = self.pdf_document[self.current_page - 1]
        
        # Render page to an image
        pix = page.get_pixmap(matrix=fitz.Matrix(self.zoom_factor, self.zoom_factor))
        img_data = pix.tobytes("ppm")
        
        # Convert to PhotoImage
        img = Image.open(io.BytesIO(img_data))
        photo = ImageTk.PhotoImage(img)
        
        # Store reference to prevent garbage collection
        self.current_photo = photo
        
        # Display image on canvas
        self.pdf_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        
        # Configure canvas scrollregion to match the image size
        self.pdf_canvas.config(scrollregion=self.pdf_canvas.bbox(tk.ALL))
        
        # Update page counter
        self.page_var.set(f"Page: {self.current_page} / {self.pdf_document.page_count}")
    
    def prev_page(self):
        """Go to the previous page"""
        if self.pdf_document and self.current_page > 1:
            self.current_page -= 1
            self.update_page_display()
    
    def next_page(self):
        """Go to the next page"""
        if self.pdf_document and self.current_page < self.pdf_document.page_count:
            self.current_page += 1
            self.update_page_display()
    
    def zoom_in(self):
        """Zoom in the PDF display"""
        self.zoom_factor *= 1.2
        self.update_page_display()
    
    def zoom_out(self):
        """Zoom out the PDF display"""
        self.zoom_factor /= 1.2
        self.update_page_display()
    
    def update_status(self, message):
        """Update the status bar message"""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About",
            "FDD Header Quality Control System\n\n"
            "A tool for verifying and correcting page numbers in FDD header extractions."
        )
    
    def show_help_batch(self):
        """Show help dialog for batch review mode"""
        help_text = (
            "FDD Header QC Review - Help\n\n"
            "This application automatically reviews files flagged during the initial batch verification.\n\n"
            "Workflow:\n"
            "1. The app automatically loads the first uncorrected file on startup.\n"
            "2. Review the items in the table (especially those marked Yellow or Red).\n"
            "3. Double-click a header or use 'Go to...' buttons to navigate the PDF.\n"
            "4. Correct the 'Expected Page' in the detail panel if necessary.\n"
            "5. Click 'Approve' for correct items or 'Reject' for incorrect ones.\n"
            "   (Approve/Reject updates the status and saves the current Expected Page value).\n"
            "6. When finished reviewing a file, click 'Save Corrected JSON'.\n"
            "7. The next uncorrected file will automatically load after saving.\n\n"
            "PDF Navigation:\n"
            "- Use the scrollbars or mouse wheel to scroll vertically and horizontally.\n"
            "- Scrolling past the bottom/top of a page automatically loads the next/previous page.\n"
            "- You can still use the Previous/Next buttons for page navigation.\n"
            "- Use Zoom In/Out buttons to adjust the view.\n\n"
            "Manual Control Options:\n"
            "- Use File > Load Flagged Pair... to select a specific file to review.\n"
            "- Use File > Load Next Uncorrected to manually load the next file.\n\n"
            "Color coding in the table:\n"
            "- Green: Verified or likely correct\n"
            "- Yellow: Needs review\n"
            "- Red: Likely incorrect\n"
            "- Gray: Not found"
        )
        messagebox.showinfo("Help", help_text)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling in the PDF viewer"""
        if not self.pdf_document:
            return
            
        # Different event details based on OS and mouse hardware
        if event.num == 4 or event.delta > 0:  # Scroll up
            self.pdf_canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:  # Scroll down
            self.pdf_canvas.yview_scroll(1, "units")
        
        # Check if we need to load the next/previous page
        current_view = self.pdf_canvas.yview()
        
        # If we're at the bottom of the page and scrolling down, go to next page
        if current_view[1] >= 1.0 and (event.num == 5 or event.delta < 0):
            if self.current_page < self.pdf_document.page_count:
                self.current_page += 1
                self.update_page_display()
                # Reset scroll to top of new page
                self.pdf_canvas.yview_moveto(0)
                
        # If we're at the top of the page and scrolling up, go to previous page
        elif current_view[0] <= 0.0 and (event.num == 4 or event.delta > 0):
            if self.current_page > 1:
                self.current_page -= 1
                self.update_page_display()
                # Set scroll to bottom of new page
                self.pdf_canvas.yview_moveto(1.0)

    def on_header_enter(self, event):
        """Handle Enter key press on a header in the table - approves the selected header"""
        self.approve_header()
