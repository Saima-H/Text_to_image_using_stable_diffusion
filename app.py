import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance, ImageTk
from customtkinter import CTkImage
from authtoken import auth_token

import torch
import os
import uuid
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler

import speech_recognition as sr
from tkinter import messagebox, filedialog, simpledialog

# Optimize CPU usage
torch.set_num_threads(os.cpu_count())

# Global variable to store last image
last_generated_image = None

# GUI Setup
app = tk.Tk()
app.geometry("1200x760")
app.title("Stable Bud - Creative Studio")
ctk.set_appearance_mode("dark")

# Prompt Entry
prompt = ctk.CTkEntry(master=app, height=40, width=500, font=("Arial", 20), text_color="black", fg_color="white")
prompt.place(x=20, y=10)

# Prompt Enhancer Function
def enhance_prompt(base_prompt, style):
    adjectives = {
        "Photorealistic": "highly detailed, sharp, realistic",
        "Anime": "anime-style, cel-shaded, vibrant",
        "Sketch": "hand-drawn, pencil sketch, black and white",
        "Fantasy": "magical, surreal, glowing"
    }
    return f"{adjectives[style]}, {base_prompt}"

# Style Dropdown
style_var = tk.StringVar(value="Photorealistic")
style_menu = ctk.CTkOptionMenu(app, values=["Photorealistic", "Anime", "Sketch", "Fantasy"], variable=style_var)
style_menu.place(x=540, y=10)

# Negative Prompt Entry
negative_prompt = ctk.CTkEntry(master=app, height=30, width=300, placeholder_text="Negative Prompt", fg_color="white", text_color="black")
negative_prompt.place(x=20, y=50)

negative_label = ctk.CTkLabel(app, text="Negative Prompt", text_color="black")
negative_label.place(x=330, y=50)

# Seed Entry
seed_var = tk.StringVar()
seed_entry = ctk.CTkEntry(app, height=30, width=120, textvariable=seed_var, fg_color="white", text_color="black")
seed_entry.place(x=450, y=50)

seed_label = ctk.CTkLabel(app, text="Seed", text_color="black")
seed_label.place(x=580, y=50)

# Blur Checkbox + Slider
blur_var = tk.BooleanVar()
blur_checkbox = ctk.CTkCheckBox(app, text="Blur", variable=blur_var, text_color="black")
blur_checkbox.place(x=20, y=90)

blur_amount = ctk.CTkSlider(app, from_=0, to=10, width=200)
blur_amount.set(0.0)
blur_amount.place(x=140, y=90)

# Contrast, Brightness, Sharpness
contrast_slider = ctk.CTkSlider(app, from_=0.5, to=2.0, width=200)
contrast_slider.set(1.0)
contrast_slider.place(x=140, y=120)

brightness_slider = ctk.CTkSlider(app, from_=0.5, to=2.0, width=200)
brightness_slider.set(1.0)
brightness_slider.place(x=140, y=150)

sharpness_slider = ctk.CTkSlider(app, from_=0.5, to=2.0, width=200)
sharpness_slider.set(1.0)
sharpness_slider.place(x=140, y=180)

contrast_label = ctk.CTkLabel(app, text="Contrast", text_color="black")
contrast_label.place(x=20, y=120)
brightness_label = ctk.CTkLabel(app, text="Brightness", text_color="black")
brightness_label.place(x=20, y=150)
sharpness_label = ctk.CTkLabel(app, text="Sharpness", text_color="black")
sharpness_label.place(x=20, y=180)

# Image Display
lmain = ctk.CTkLabel(master=app, height=512, width=512)
lmain.place(x=20, y=230)

# Gallery Panel
gallery_frame = tk.Frame(app, bg="#222222", width=450, height=650)
gallery_frame.place(x=580, y=90)

gallery_title = tk.Label(gallery_frame, text="Generated Images", font=("Arial", 16), bg="#222222", fg="white")
gallery_title.pack(pady=10)

gallery_canvas = tk.Canvas(gallery_frame, bg="#222222", width=430, height=580)
gallery_canvas.pack(side=tk.LEFT, fill=tk.BOTH)

gallery_thumbnails = []

# Load Models
modelid = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(modelid, torch_dtype=torch.float32, use_auth_token=auth_token).to("cpu")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(modelid, torch_dtype=torch.float32, use_auth_token=auth_token).to("cpu")
inpaint_pipe.scheduler = DPMSolverMultistepScheduler.from_config(inpaint_pipe.scheduler.config)

# 🎤 Voice Input (Long Prompt Friendly)
def listen_to_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        messagebox.showinfo("🎤 Listening", "Speak your prompt now...")
        audio = recognizer.listen(source, timeout=30)
    try:
        text = recognizer.recognize_google(audio)
        prompt.delete(0, tk.END)
        prompt.insert(0, text)
    except sr.UnknownValueError:
        messagebox.showerror("Error", "Could not understand audio.")
    except sr.RequestError:
        messagebox.showerror("Error", "Could not access speech service.")

# Generate Image
def generate():
    global last_generated_image
    base_prompt = prompt.get()
    if not base_prompt.strip():
        messagebox.showwarning("Empty Prompt", "Please type or speak a prompt first.")
        return

    style = style_var.get()
    full_prompt = enhance_prompt(base_prompt, style)
    neg_prompt = negative_prompt.get() or "low quality, distorted, deformed"

    seed_val = seed_var.get()
    generator = None
    if seed_val.isdigit():
        generator = torch.manual_seed(int(seed_val))

    image = pipe(
        full_prompt,
        guidance_scale=8.0,
        negative_prompt=neg_prompt,
        num_inference_steps=15,
        height=512,
        width=512,
        generator=generator
    ).images[0]

    if blur_var.get():
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_amount.get()))

    image = ImageEnhance.Contrast(image).enhance(contrast_slider.get())
    image = ImageEnhance.Brightness(image).enhance(brightness_slider.get())
    image = ImageEnhance.Sharpness(image).enhance(sharpness_slider.get())

    filename = f"generated_{uuid.uuid4().hex[:6]}.png"
    image.save(filename)
    last_generated_image = image

    img = CTkImage(light_image=image, size=(512, 512))
    lmain.configure(image=img)
    lmain.image = img

    add_to_gallery(image, filename)

# Add to Gallery
def add_to_gallery(img, filename):
    img_thumb = img.copy().resize((100, 100))
    img_tk = CTkImage(light_image=img_thumb, size=(100, 100))
    label = ctk.CTkLabel(master=gallery_canvas, image=img_tk, text="")
    label.image = img_tk
    label.pack(pady=5)
    gallery_thumbnails.append((label, filename))

# Region-based Edit with Brush
def open_mask_editor():
    if last_generated_image is None:
        messagebox.showerror("No Image", "Generate an image before editing.")
        return

    editor = tk.Toplevel(app)
    editor.title("Mask Editor - Paint Area to Edit")
    editor.geometry("512x512")
    canvas = tk.Canvas(editor, width=512, height=512, bg="white")
    canvas.pack()

    mask_image = Image.new("L", (512, 512), 255)
    draw = ImageDraw.Draw(mask_image)

    tk_img = ImageTk.PhotoImage(last_generated_image.resize((512, 512)))
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
    canvas.image = tk_img

    def paint(event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        draw.ellipse([x1, y1, x2, y2], fill=0)

    canvas.bind("<B1-Motion>", paint)

    def confirm_edit():
        editor.destroy()
        do_inpaint_with_mask(mask_image)

    confirm_btn = ctk.CTkButton(editor, text="Apply Edit", command=confirm_edit)
    confirm_btn.place(x=180, y=460)

# Inpaint using drawn mask
def do_inpaint_with_mask(mask_image):
    edit_prompt = simpledialog.askstring("Edit Prompt", "Enter new prompt for masked region:")
    if not edit_prompt:
        return

    mask_resized = mask_image.resize((512, 512)).convert("RGB")
    edited = inpaint_pipe(
        prompt=edit_prompt,
        image=last_generated_image,
        mask_image=mask_resized,
        guidance_scale=8.0,
        num_inference_steps=15
    ).images[0]

    filename = f"edited_{uuid.uuid4().hex[:6]}.png"
    edited.save(filename)

    img = CTkImage(light_image=edited, size=(512, 512))
    lmain.configure(image=img)
    lmain.image = img
    add_to_gallery(edited, filename)
    messagebox.showinfo("Success", "Region-based edit complete.")

# Buttons
voice_btn = ctk.CTkButton(master=app, height=40, width=120, font=("Arial", 16), text_color="white", fg_color="green", text="🎙️ Speak", command=listen_to_voice)
voice_btn.place(x=20, y=200)

gen_btn = ctk.CTkButton(master=app, height=40, width=120, font=("Arial", 16), text_color="white", fg_color="blue", text="Generate", command=generate)
gen_btn.place(x=160, y=200)

edit_btn = ctk.CTkButton(master=app, height=40, width=140, font=("Arial", 16), text_color="white", fg_color="purple", text="🩹 Edit with Mask", command=open_mask_editor)
edit_btn.place(x=300, y=200)

app.mainloop()