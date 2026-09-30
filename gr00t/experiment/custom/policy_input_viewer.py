"""Display the language and RGB frame history supplied to a rollout policy."""

from collections.abc import Sequence

import numpy as np


class PolicyInputViewer:
    """Small companion window; GUI dependencies are loaded only when opened."""

    def __init__(
        self,
        camera_name: str | Sequence[str],
        video_delta_indices: Sequence[int],
        eef_overlay: bool = False,
        joint_overlay: bool = False,
        rotation_overlay: bool = False,
    ):
        import tkinter as tk

        from PIL import Image, ImageTk

        self._Image = Image
        self._ImageTk = ImageTk
        self.delta_indices = tuple(video_delta_indices)
        self.camera_names = (camera_name,) if isinstance(camera_name, str) else tuple(camera_name)
        if not self.camera_names or not self.delta_indices:
            raise ValueError("At least one camera and frame offset are required")
        self.closed = False
        self._shown = False
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title("GR00T policy input")
        self.root.configure(background="#111820")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        columns = min(3, len(self.delta_indices) * len(self.camera_names))
        self.tile_width = min(480, (self.root.winfo_screenwidth() - 80) // columns)
        width = columns * self.tile_width
        text_style = {"background": "#111820", "foreground": "#edf2f7", "anchor": "w"}
        self.status = tk.Label(self.root, font=("TkDefaultFont", 11), **text_style)
        self.status.pack(fill="x", padx=16, pady=(12, 4))
        tk.Label(
            self.root, text="Language command", font=("TkDefaultFont", 11, "bold"), **text_style
        ).pack(fill="x", padx=16)
        self.instruction = tk.Label(
            self.root,
            font=("TkDefaultFont", 14),
            justify="left",
            wraplength=width,
            **text_style,
        )
        self.instruction.pack(fill="x", padx=16, pady=(4, 12))
        tk.Label(self.root, text=f"Cameras: {', '.join(self.camera_names)}", **text_style).pack(
            fill="x", padx=16
        )

        grid = tk.Frame(self.root, background="#111820")
        grid.pack(padx=12, pady=8)
        self.image_labels = []
        tiles = [(camera, delta) for camera in self.camera_names for delta in self.delta_indices]
        for index, (camera, delta) in enumerate(tiles):
            cell = tk.Frame(grid, background="#111820")
            cell.grid(row=index // columns, column=index % columns, padx=4, pady=4)
            title = "Current frame (t=0)" if delta == 0 else f"History frame (t={delta})"
            if len(self.camera_names) > 1:
                title = f"{camera}\n{title}"
            tk.Label(cell, text=title, **text_style).pack(fill="x", pady=(0, 4))
            label = tk.Label(cell, background="#111820", borderwidth=0)
            label.pack()
            self.image_labels.append(label)

        tk.Label(
            self.root,
            text="Images refresh when a new observation is supplied to the policy.",
            **text_style,
        ).pack(fill="x", padx=16, pady=(0, 12))
        if eef_overlay:
            tk.Label(
                self.root,
                text=(
                    "Simulator: cyan = L EEF target, magenta = R EEF target, white = current wrist.\n"
                    "Lines show position error. Robot motion follows joint commands."
                ),
                justify="left",
                **text_style,
            ).pack(fill="x", padx=16, pady=(0, 12))
        if joint_overlay:
            tk.Label(
                self.root,
                text=(
                    "Translucent arms: joint-command poses (cyan = L, magenta = R).\n"
                    "Includes waist/hand targets and joint-limit clipping."
                ),
                justify="left",
                **text_style,
            ).pack(fill="x", padx=16, pady=(0, 12))
        if eef_overlay and rotation_overlay:
            tk.Label(
                self.root,
                text=(
                    "Wrist axes: X red / Y green / Z blue. Long = EEF, short = actual, medium = joint FK.\n"
                    "Degrees compare EEF with actual wrist / joint-command FK. Frame: wrist_yaw_link."
                ),
                justify="left",
                wraplength=width,
                **text_style,
            ).pack(fill="x", padx=16, pady=(0, 12))

    def update(
        self,
        instruction: str,
        frames: np.ndarray | dict[str, np.ndarray],
        *,
        episode: int,
        step: int,
        smoke_test: bool,
    ):
        if self.closed:
            return
        if isinstance(frames, np.ndarray):
            frames = {self.camera_names[0]: frames}
        images = []
        for camera in self.camera_names:
            history = frames[camera]
            if (
                history.ndim != 4
                or history.shape[0] != len(self.delta_indices)
                or history.shape[-1] != 3
            ):
                raise ValueError(
                    f"Expected RGB history for {camera} at {self.delta_indices}, got {history.shape}"
                )
            images.extend(history)
        mode = "SMOKE TEST (no policy request)" if smoke_test else "POLICY INPUT"
        self.status.configure(text=f"{mode}  |  Episode {episode}  |  Step {step}")
        self.instruction.configure(text=instruction)
        for label, frame in zip(self.image_labels, images, strict=True):
            # Observations are already upright RGB. Do not flip them again or
            # convert to OpenCV's BGR convention. Resize only the display copy.
            image = self._Image.fromarray(frame)
            image.thumbnail((self.tile_width, self.tile_width), self._Image.Resampling.LANCZOS)
            photo = self._ImageTk.PhotoImage(image, master=self.root)
            label.configure(image=photo)
            label.image = photo  # Tk does not retain the Python image object.
        if not self._shown:
            self.root.deiconify()
            self._shown = True
        self.poll()

    def poll(self):
        if not self.closed:
            self.root.update_idletasks()
            self.root.update()

    def close(self):
        if not self.closed:
            self.closed = True
            self.root.destroy()
