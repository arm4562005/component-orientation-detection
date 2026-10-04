import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from datetime import datetime

import cv2
from PIL import Image, ImageTk

from config import (
    REFERENCE_FILE,
    DEBUG_DIR,
    DEBUG_REFERENCE_DIR,
    DEBUG_INSPECTIONS_DIR,
    DEBUG_LATEST_DIR,
    WINDOW_GEOMETRY,
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT,
)

from registration import FixtureRegistration
from detector import ComponentOrientationDetector

from visualization import (
    draw_inspection,
    draw_reference_discovery,
    draw_stage_component_detection,
    draw_registration_stage,
    draw_comparison,
    save_debug,
)


                                                
BG = "#F4F7FB"
CARD = "#FFFFFF"
BORDER = "#E2E8F0"
TEXT = "#1E293B"
MUTED = "#64748B"
PRIMARY = "#2563EB"
PRIMARY_HOVER = "#1D4ED8"
SUCCESS = "#16A34A"
SUCCESS_HOVER = "#15803D"
DANGER = "#DC2626"
NEUTRAL_BTN = "#F1F5F9"
NEUTRAL_BORDER = "#CBD5E1"


class OrientationApp:

    def __init__(self, root):

        self.root = root
        self.root.title("Component Orientation Detection")
        self.root.geometry("1280x820")
        self.root.minsize(1000, 720)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

                                                           
                      
                                                           
        self.camera = None
        self.camera_running = False
        self.last_frame = None
        self._display_source = None                                     

                                                           
                                    
                                                           
        self.registration = None
        self.detector = None
        self.reference_ready = False

        self.last_inspection = None
        self.last_inspected_image = None

                                                           
                      
                                                           
        self.status_var = tk.StringVar(value="Initializing system...")
        self.components_var = tk.StringVar(value="Components Detected: —")
        self.levers_var = tk.StringVar(value="Levers Detected: —")
        self.result_var = tk.StringVar(value="Result: Not Inspected")
        self.debug_var = tk.BooleanVar(value=False)
        self.compare_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._register_reference()

                                                                              
        self.root.bind("<Configure>", self._on_window_resize)
        self._resize_job = None

                                                           
                                                    
                                                           

    def _build_ui(self):
                            
        header = tk.Frame(self.root, bg=BG, height=56)
        header.pack(fill="x", padx=32, pady=(20, 8))
        header.pack_propagate(False)

        left_h = tk.Frame(header, bg=BG)
        left_h.pack(side="left", fill="y")

        accent = tk.Canvas(left_h, width=12, height=12, bg=BG, highlightthickness=0)
        accent.pack(side="left", pady=20)
        accent.create_oval(0, 0, 12, 12, fill=PRIMARY, outline="")

        bar = tk.Frame(left_h, bg=PRIMARY, width=3, height=16)
        bar.pack(side="left", padx=(10, 12), pady=18)

        tk.Label(
            left_h,
            text="COMPONENT ORIENTATION DETECTION",
            font=("Segoe UI", 15, "bold"),
            fg=TEXT,
            bg=BG,
        ).pack(side="left", pady=14)

                                                
        content = tk.Frame(self.root, bg=BG)
        content.pack(fill="both", expand=True, padx=32, pady=(4, 8))

                                                   
        left_card = tk.Frame(
            content,
            bg=CARD,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        left_card.pack(side="left", fill="both", expand=True, padx=(0, 16))

        self.image_inner = tk.Frame(left_card, bg="#F1F5F9")
        self.image_inner.pack(fill="both", expand=True, padx=12, pady=12)

        self.image_label = tk.Label(
            self.image_inner,
            text="No Camera Feed\n\nStart the camera or upload an image",
            font=("Segoe UI", 12),
            fg=MUTED,
            bg="#F1F5F9",
            justify="center",
        )
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")

                      
        right = tk.Frame(content, bg=BG, width=340)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        right.pack_configure(pady=(0, 0))

                      
        results_card = tk.Frame(
            right,
            bg=CARD,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        results_card.pack(side="top", fill="both", expand=True, pady=(0, 12))

        res_header = tk.Frame(results_card, bg=CARD)
        res_header.pack(fill="x", padx=16, pady=(14, 6))

        tk.Label(
            res_header,
            text="Detection Results:",
            font=("Segoe UI", 11, "bold"),
            fg=TEXT,
            bg=CARD,
        ).pack(side="left")

        self.badge = tk.Label(
            res_header,
            textvariable=self.components_var,
            font=("Segoe UI", 9),
            fg=PRIMARY,
            bg="#EFF6FF",
            padx=8,
            pady=2,
        )
        self.badge.pack(side="right")

                                                           
        self.empty_label = tk.Label(
            results_card,
            text="No inspection yet\n\nUpload an image or start camera,\nthen press Inspect.",
            font=("Segoe UI", 10),
            fg=MUTED,
            bg=CARD,
            justify="center",
        )
        self.empty_label.pack(fill="both", expand=True, padx=12, pady=24)

                                                                 
                                                                           
                                                                            
        self.results_view = tk.Frame(results_card, bg=CARD)

        self.results_canvas = tk.Canvas(
            self.results_view,
            bg=CARD,
            highlightthickness=0,
            bd=0,
            yscrollincrement=2,
        )
        self.results_scrollbar = ttk.Scrollbar(
            self.results_view,
            orient="vertical",
            command=self.results_canvas.yview,
        )
        self.results_canvas.configure(yscrollcommand=self.results_scrollbar.set)

        self.results_scrollbar.pack(side="right", fill="y")
        self.results_canvas.pack(side="left", fill="both", expand=True)

        self.results_body = tk.Frame(self.results_canvas, bg=CARD)
        self._results_window = self.results_canvas.create_window(
            (0, 0), window=self.results_body, anchor="nw"
        )

        self.results_body.bind(
            "<Configure>",
            lambda e: self._update_results_scrollregion(),
        )
        self.results_canvas.bind(
            "<Configure>",
            lambda e: self.results_canvas.itemconfigure(
                self._results_window, width=e.width
            ),
        )

                                                       
        self.results_canvas.bind("<Enter>", self._bind_results_mousewheel)
        self.results_canvas.bind("<Leave>", self._unbind_results_mousewheel)

                                                                                  
        self.cards_frame = tk.Frame(self.results_body, bg=CARD)
        self.cards_frame.pack(fill="x", padx=8, pady=(4, 2))

        self.comp_cards = []
        self._selected_comp = None
        self._last_results_by_id = {}

        for i in range(10):
            cid = i + 1

                                                                    
            card = tk.Frame(
                self.cards_frame,
                bg="#FFFFFF",
                highlightbackground="#E2E8F0",
                highlightthickness=1,
                height=44,
                cursor="hand2",
            )
            card.pack(
                fill="x",
                pady=2,
                padx=2,
            )
            card.pack_propagate(False)

                                                               
            inner = tk.Frame(
                card,
                bg="#FFFFFF",
                highlightbackground="#E2E8F0",
                highlightthickness=1,
                cursor="hand2",
            )
            inner.pack(
                fill="both",
                expand=True,
                padx=2,
                pady=2,
            )

                                                                    
                                                       
                                                                    
            left = tk.Frame(
                inner,
                bg="#FFFFFF",
                cursor="hand2",
            )
            left.pack(
                side="left",
                padx=(8, 2),
            )

            num_lbl = tk.Label(
                left,
                text=f"C{cid:02d}",
                font=("Segoe UI", 9, "bold"),
                fg="#334155",
                bg="#FFFFFF",
                cursor="hand2",
                width=4,
                anchor="w",
            )
            num_lbl.pack(
                side="left"
            )

            icon_canvas = tk.Canvas(
                left,
                width=22,
                height=22,
                bg="#FFFFFF",
                highlightthickness=0,
                cursor="hand2",
            )
            icon_canvas.pack(
                side="left",
                padx=(4, 6),
            )

            status_lbl = tk.Label(
                left,
                text="—",
                font=("Segoe UI", 9, "bold"),
                fg="#64748B",
                bg="#FFFFFF",
                cursor="hand2",
                width=7,
                anchor="w",
            )
            status_lbl.pack(
                side="left"
            )

                                                                    
                                                     
                                         
                                                                    
            summary_lbl = tk.Label(
                inner,
                text="",
                font=("Segoe UI", 8, "bold"),
                fg="#64748B",
                bg="#F8FAFC",
                padx=8,
                pady=2,
                cursor="hand2",
                anchor="center",
                highlightbackground="#E2E8F0",
                highlightthickness=1,
            )
            summary_lbl.pack(
                side="right",
                padx=(8, 8),
                pady=4,
            )

            widgets = (
                card,
                inner,
                left,
                num_lbl,
                icon_canvas,
                status_lbl,
                summary_lbl,
            )

            for widget in widgets:
                widget.bind(
                    "<Button-1>",
                    lambda e, c=cid: self._select_component(c),
                )

            self.comp_cards.append(
                {
                    "frame": card,
                    "inner": inner,
                    "left": left,
                    "num": num_lbl,
                    "icon": icon_canvas,
                    "status": status_lbl,
                    "summary": summary_lbl,
                    "id": cid,
                }
            )

                                             
        self.detail_frame = tk.Frame(
            self.results_body, bg="#F8FAFC",
            highlightbackground="#E2E8F0", highlightthickness=1,
            height=165,
        )
        self.detail_frame.pack(fill="x", padx=10, pady=(6, 8))
        self.detail_frame.pack_propagate(False)
        self.detail_frame.bind(
            "<Configure>",
            lambda e: self.root.after_idle(self._update_results_scrollregion),
        )

        self.detail_title = tk.Label(
            self.detail_frame, text="Select a component",
            font=("Segoe UI", 10, "bold"), fg=TEXT, bg="#F8FAFC", anchor="w",
        )
        self.detail_title.pack(fill="x", padx=12, pady=(10, 4))

        self.detail_text = tk.Label(
            self.detail_frame,
            text="Click any component above to see detection details.",
            font=("Segoe UI", 9), fg=MUTED, bg="#F8FAFC",
            justify="left", anchor="nw",
        )
        self.detail_text.pack(fill="both", expand=True, padx=12, pady=(0, 10))

                                                                      

                            
        result_card = tk.Frame(
            right,
            bg=CARD,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        result_card.pack(side="bottom", fill="x", pady=(12, 0), before=results_card)

        result_inner = tk.Frame(result_card, bg=CARD)
        result_inner.pack(fill="x", padx=16, pady=10)

        self.result_dot = tk.Canvas(
            result_inner, width=22, height=22, bg=CARD, highlightthickness=0
        )
        self.result_dot.pack(side="left", padx=(0, 10))
        self._draw_result_dot("#94A3B8")

        self.result_label = tk.Label(
            result_inner,
            textvariable=self.result_var,
            font=("Segoe UI", 12, "bold"),
            fg=TEXT,
            bg=CARD,
            anchor="w",
        )
        self.result_label.pack(side="left", fill="x", expand=True)

                                
        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(fill="x", padx=32, pady=(4, 8))

        btn_bar = tk.Frame(bottom, bg=BG)
        btn_bar.pack(side="left")

        self._modern_button(
            btn_bar, "▶  Start Camera", self.start_camera,
            bg=PRIMARY, fg="white", active_bg=PRIMARY_HOVER
        )
        self._modern_button(
            btn_bar, "■  Stop Camera", self.stop_camera,
            bg=NEUTRAL_BTN, fg=TEXT, active_bg="#E2E8F0",
            border=NEUTRAL_BORDER
        )
        self._modern_button(
            btn_bar, "Upload Image", self.upload_image,
            bg=CARD, fg=TEXT, active_bg=NEUTRAL_BTN,
            border=NEUTRAL_BORDER
        )
        self._modern_button(
            btn_bar, "Inspect", self.inspect_current,
            bg=SUCCESS, fg="white", active_bg=SUCCESS_HOVER
        )
        self._modern_button(
            btn_bar, "Reset", self.reset_inspection,
            bg=NEUTRAL_BTN, fg=TEXT, active_bg="#E2E8F0",
            border=NEUTRAL_BORDER
        )

        opts = tk.Frame(bottom, bg=BG)
        opts.pack(side="right")

        tk.Checkbutton(
            opts,
            text="Comparison",
            variable=self.compare_var,
            font=("Segoe UI", 9),
            fg=MUTED,
            bg=BG,
            activebackground=BG,
            selectcolor=CARD,
            command=self._refresh_display,
            cursor="hand2",
        ).pack(side="left")

                     
        tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            fg=MUTED,
            bg=BG,
            anchor="w",
        ).pack(fill="x", padx=32, pady=(0, 14))

    def _modern_button(self, parent, text, command, bg, fg, active_bg, border=None):
        kwargs = dict(
            text=text,
            font=("Segoe UI", 10, "bold"),
            bg=bg,
            fg=fg,
            activebackground=active_bg,
            activeforeground=fg,
            relief="flat",
            bd=0,
            padx=16,
            pady=9,
            cursor="hand2",
            command=command,
        )
        if border:
            kwargs["highlightbackground"] = border
            kwargs["highlightthickness"] = 1
        btn = tk.Button(parent, **kwargs)
        btn.pack(side="left", padx=(0, 8))
        return btn

    def _draw_result_dot(self, color):
        self.result_dot.delete("all")
        self.result_dot.create_oval(1, 1, 21, 21, fill=color, outline="")

    def _update_results_scrollregion(self):
                                                                          
                                                                        
                                                                        
        self.root.update_idletasks()

        body_w = self.results_body.winfo_reqwidth()
        body_h = self.results_body.winfo_reqheight()
        canvas_w = self.results_canvas.winfo_width()

                                                                           
                                                                        
                    
        bottom_padding = 16
        scroll_h = body_h + bottom_padding
        scroll_w = max(body_w, canvas_w)

        self.results_canvas.configure(
            scrollregion=(0, 0, scroll_w, scroll_h)
        )

    def _bind_results_mousewheel(self, event=None):
        self.results_canvas.bind_all("<MouseWheel>", self._on_results_mousewheel)
        self.results_canvas.bind_all("<Button-4>", self._on_results_mousewheel)
        self.results_canvas.bind_all("<Button-5>", self._on_results_mousewheel)

    def _unbind_results_mousewheel(self, event=None):
        self.results_canvas.unbind_all("<MouseWheel>")
        self.results_canvas.unbind_all("<Button-4>")
        self.results_canvas.unbind_all("<Button-5>")

    def _on_results_mousewheel(self, event):
                                                            
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = -int(event.delta / 120)
        self.results_canvas.yview_scroll(delta, "units")

    def _set_results_placeholder(self):
                                                                        
        self.results_view.pack_forget()
        self.empty_label.pack(fill="both", expand=True, padx=12, pady=24)
        self._selected_comp = None
        self._last_results_by_id = {}

                                                            
        self.results_canvas.yview_moveto(0)

                                                           
                                               
                                                           

    def _register_reference(self):
        try:
            if not REFERENCE_FILE.exists():
                raise RuntimeError(
                    "REFERENCE REGISTRATION FAILED\n\n"
                    f"Missing reference file:\n{REFERENCE_FILE}"
                )

            self.status_var.set("Registering known-good reference...")
            self.root.update_idletasks()

            self.registration = FixtureRegistration()
            self.detector = ComponentOrientationDetector(self.registration)

            if len(self.detector.reference_components) != 10:
                raise RuntimeError(
                    "REFERENCE REGISTRATION FAILED\n\n"
                    "The detector did not produce exactly 10 reference components."
                )

            for comp in self.detector.reference_components:
                if (
                    comp.lever_contour is None
                    or comp.pivot is None
                    or comp.tip is None
                ):
                    raise RuntimeError(
                        "REFERENCE REGISTRATION FAILED\n\n"
                        f"Component {comp.component_id:02d} has an invalid lever measurement."
                    )

            self.reference_ready = True
            self.status_var.set(
                "Ready — Reference calibrated: 10/10 components and 10/10 levers."
            )
            self.components_var.set("Components Detected: —")
            self.levers_var.set("Levers Detected: —")
            self.result_var.set("Result: Not Inspected")
            self._draw_result_dot("#94A3B8")
            self._show_waiting_screen()

            reference_debug = draw_reference_discovery(
                self.registration.reference,
                self.detector.reference_components,
            )
            save_debug(
                DEBUG_REFERENCE_DIR,
                {
                    "01_reference_original.jpg": self.registration.reference,
                    "02_reference_discovery.jpg": reference_debug,
                },
            )

        except Exception as exc:
            self.reference_ready = False
            self.status_var.set("REFERENCE REGISTRATION FAILED")
            self.result_var.set("Result: Reject / Alert")
            self._draw_result_dot(DANGER)
            self.components_var.set("Components Detected: —")
            self.levers_var.set("Levers Detected: —")
            detail = str(exc).strip() or "Unknown startup calibration error."
            self._show_error_screen(
                "REFERENCE REGISTRATION FAILED\n\n" + detail
            )
            messagebox.showerror("Reference Registration Failed", detail)

    def _show_waiting_screen(self):
        self.image_inner.configure(bg="#F1F5F9")
        try:
            self.image_label.pack_forget()
        except Exception:
            pass
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")
        self.image_label.configure(
            image="",
            text="No Camera Feed\n\nStart the camera or upload an image",
            bg="#F1F5F9",
            fg=MUTED,
            font=("Segoe UI", 12),
            justify="center",
        )
        self.image_label.image = None

    def _show_error_screen(self, message):
        self.image_inner.configure(bg="#FEF2F2")
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")
        self.image_label.configure(
            image="",
            text=f"{message}\n\nInspection is disabled.",
            bg="#FEF2F2",
            fg=DANGER,
            font=("Segoe UI", 13, "bold"),
            justify="center",
        )
        self.image_label.image = None

                                                           
                   
                                                           

    def show_image(self, image):
        if image is None or image.size == 0:
            return

        self._display_source = image

                                                                         
                                                                             
        self.root.update_idletasks()
        panel_w = max(self.image_inner.winfo_width() - 8, 1)
        panel_h = max(self.image_inner.winfo_height() - 8, 1)

        if panel_w < 50 or panel_h < 50:
            panel_w = DISPLAY_WIDTH
            panel_h = DISPLAY_HEIGHT

        h, w = image.shape[:2]
        scale = min(panel_w / max(w, 1), panel_h / max(h, 1))
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))

        display = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(display))

                                                                 
        self.image_inner.configure(bg=CARD)
        self.image_label.place_forget()
        self.image_label.pack(fill="both", expand=True)
        self.image_label.configure(image=photo, text="", bg=CARD)
        self.image_label.image = photo

    def _on_window_resize(self, event):
                                                                       
        if event.widget is not self.root:
            return
        if self._display_source is None:
            return
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(120, self._refit_display)

    def _refit_display(self):
        self._resize_job = None
        if self._display_source is not None:
            self.show_image(self._display_source)

    def _refresh_display(self):
        if self.last_inspection is None or self.last_inspected_image is None:
            return

        if self.compare_var.get():
            combined = draw_comparison(
                self.registration.reference,
                self.detector.reference_components,
                self.last_inspected_image,
                self.last_inspection,
            )
            self.show_image(combined)
        else:
            annotated = draw_inspection(
                self.last_inspected_image,
                self.last_inspection,
            )
            self.show_image(annotated)

                                                           
                   
                                                           

    def _build_display_map(self, inspection):
        """
        Build the SAME current-image numbering used by visualization.py.

        Internal result['id'] values remain the calibrated physical reference
        IDs. display_id is UI-only and follows what the operator sees:

            horizontal: top row C01..C05, bottom row C06..C10
            vertical:   left column C01..C05, right column C06..C10

        This keeps the result panel and the annotated image synchronized.
        """
        try:
            from visualization import get_current_display_map
            return get_current_display_map(inspection)
        except Exception:
            return {
                index: int(result.get("id", index + 1))
                for index, result in enumerate(inspection.results)
            }

                                                           
                   
                                                           

    def _write_results(self, inspection):
                                                                 
        self.empty_label.pack_forget()
        self.results_view.pack(
            fill="both",
            expand=True,
            padx=(8, 4),
            pady=(0, 4),
        )
        self.results_canvas.yview_moveto(0)

        if inspection is None or not inspection.success:
            self._last_results_by_id = {}

            for card in self.comp_cards:
                self._style_card(
                    card,
                    "FAIL",
                    "#F8FAFC",
                    "#202020",
                    "#DC2626",
                )

            self.detail_title.configure(
                text="Registration failed"
            )
            self.detail_text.configure(
                text=(
                    "Fixture could not be registered.\n"
                    "No component details available."
                )
            )
            return

                                                            
                                                               
                                                  
                                                            
        display_map = self._build_display_map(
            inspection
        )

        self._last_results_by_id = {}

        for index, result in enumerate(
            inspection.results
        ):
            display_id = int(
                display_map.get(
                    index,
                    result.get(
                        "id",
                        index + 1,
                    ),
                )
            )

                               
            result["display_id"] = display_id

            self._last_results_by_id[
                display_id
            ] = result

                                                            
                                                
                                                            
        for card in self.comp_cards:
            cid = int(
                card["id"]
            )

            result = self._last_results_by_id.get(
                cid
            )

            status = (
                result.get(
                    "status",
                    "UNKNOWN",
                )
                if result
                else "UNKNOWN"
            )

            if status == "CORRECT":
                self._style_card(
                    card,
                    "OK",
                    "#F8FAFC",
                    "#202020",
                    "#2E8B57",
                )

            elif status == "WRONG ORIENTATION":
                self._style_card(
                    card,
                    "WRONG",
                    "#F8FAFC",
                    "#202020",
                    "#DC2626",
                )

            else:
                self._style_card(
                    card,
                    "N/A",
                    "#F8FAFC",
                    "#202020",
                    "#6B7280",
                )

                                                                   
        prefer = next(
            (
                r.get(
                    "display_id",
                    r.get("id", 1),
                )
                for r in inspection.results
                if r.get("status") != "CORRECT"
            ),
            1,
        )

        self._select_component(
            int(prefer)
        )

        self.root.after_idle(
            self._update_results_scrollregion
        )

    def _style_card(
        self,
        card,
        label,
        color,
        bg,
        border_color,
        selected=False,
    ):
        """
        Light UI result row.

        The actual inspection logic is untouched.
        This function only controls how the result is displayed.
        """

        card_bg = "#FFFFFF"

                                                                    
        outer_border = (
            border_color
            if selected
            else "#E2E8F0"
        )

        outer_thickness = (
            2
            if selected
            else 1
        )

        card["frame"].configure(
            bg=card_bg,
            highlightbackground=outer_border,
            highlightthickness=outer_thickness,
        )

        card["inner"].configure(
            bg=card_bg,
            highlightbackground="#E2E8F0",
            highlightthickness=1,
        )

        card["left"].configure(
            bg=card_bg
        )

        card["num"].configure(
            bg=card_bg,
            fg="#334155",
        )

                                                       
                          
                                                       
        if label == "OK":
            status_color = "#15803D"
        elif label in ("WRONG", "FAIL"):
            status_color = "#DC2626"
        else:
            status_color = "#64748B"

        card["status"].configure(
            bg=card_bg,
            fg=status_color,
            text=label,
        )

                                                       
                            
                                                       
        if label == "OK":
            summary = "✓ Verified"
            summary_fg = "#15803D"
            summary_bg = "#F0FDF4"
            summary_border = "#BBF7D0"

        elif label in ("WRONG", "FAIL"):
            summary = "⚠ Check Part"
            summary_fg = "#B91C1C"
            summary_bg = "#FEF2F2"
            summary_border = "#FECACA"

        else:
            summary = "— Unverified"
            summary_fg = "#64748B"
            summary_bg = "#F8FAFC"
            summary_border = "#CBD5E1"

        card["summary"].configure(
            text=summary,
            bg=summary_bg,