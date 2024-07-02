from turtle import st, width
import warnings

from networkx import radius
from numpy import pad
warnings.filterwarnings('ignore') # ignore warnings
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

# Your existing code here
import tkinter as tk
import customtkinter as ctk 
from PIL import Image,ImageTk
# from authtoken import auth_token
import torch
from torch import autocast, device
# from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler  
from diffusers import StableDiffusion3Pipeline 
from transformers import BlipProcessor, BlipForConditionalGeneration
# from transformers import T5EncoderModel, BitsAndBytesConfig #(sd3 model)

# code downloads the model in users/.cache/hugging face folder

# Create the app
app = ctk.CTk()
app.geometry("600x800")
app.title("VisionaryScribe")     
app.set_appearance_mode("dark")    
# app.configure(bg='gray20')


# Pipelines

device = "cuda"
# model_weight_path = 'ema-pruned.ckpt'

# SD 2-1 model pipeline 
# *********************

# model_id = "stabilityai/stable-diffusion-2-1" #"runwayml/stable-diffusion-v1-5","CompVis/stable-diffusion-v1-4"(other model to import)

# pipe = StableDiffusionPipeline.from_pretrained(
#     model_id, 
#     variant="fp16",  # earlier model uses 'revision'
#     torch_dtype=torch.float16
#     ) 
# # use_auth_token=auth_token, checkpoint_path = model_weight_path(not working)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


# Stable Diffusion Pipeline 3
# ***************************
model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
blip_model_name = "Salesforce/blip-image-captioning-large"  

# torch.set_float32_matmul_precision("high")


# torch._inductor.config.conv_1x1_as_mm = True
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.epilogue_fusion = False
# torch._inductor.config.coordinate_descent_check_all_directions = True

pipe = StableDiffusion3Pipeline.from_pretrained( 
    model_id,
    text_encoder_3=None,
    tokenizer_3=None,
    torch_dtype=torch.float16
).to("cuda")


# pipe.set_progress_bar_config(disable=True)
# pipe.transformer.to(memory_format=torch.channels_last)
# pipe.vae.to(memory_format=torch.channels_last)
# pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
# pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)



# weight file checker
print(f"Using model weights from: {model_id} app successfully loaded")


# BLIP model for captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")
#ADD ON 
print(f"Using model weights from: {blip_model_name} app successfully loaded")



guidance_scale=7.0  # default value
image_height = 256
image_width = 256
inference_steps = 10 

def move_cursor_word_right(event):
    text_var = prompt.get()  # Assuming prompt is linked to a StringVar
    current_text = text_var
    current_position = len(current_text)
    new_position = current_text.find(' ', current_position + 1)
    if new_position == -1:
        new_position = len(current_text)


def move_cursor_word_left(event):
    text_var = prompt.get()  # Assuming prompt is linked to a StringVar
    current_text = text_var
    # Simulate cursor position based on text length (not accurate)
    current_position = len(current_text)
    new_position = current_text.rfind(' ', 0, current_position - 1)
    if new_position == -1:
        new_position = 0

def clear_prompt():
    prompt.delete(0, tk.END)



def update_guidance_scale(value):
    global guidance_scale
    guidance_scale = float(value)
    guidance_scale_label.configure(text=f"Guidance Scale: {guidance_scale:.2f}")
    # print(f"Guidance Scale updated to: {guidance_scale}")

def optionmenu_callback(choice):
    global image_height, image_width
    print("optionmenu dropdown clicked:", choice)
    if choice == "512*512":
        image_height, image_width = 512, 512
    elif choice == "512*720":
        image_height, image_width = 512, 720
    elif choice == "512*1024":
        image_height, image_width = 512, 1024
    elif choice == "720*512":
        image_height, image_width = 720, 512
    elif choice == "720*720":
        image_height, image_width = 720, 720
    elif choice == "720*1024":
        image_height, image_width = 720, 1024
    elif choice == "1024*512":
        image_height, image_width = 1024, 512
    elif choice == "1024*720":
        image_height, image_width = 1024, 720
    elif choice == "1024*1024":
        image_height, image_width = 1024, 1024
    # Add more options as needed


# Modify the generate    

def generate():     
# Generate the image
    with autocast(device):  
        global guidance_scale, image_height, image_width
        seed_value = seed_value_entry.get()  # Assuming `seed_value_entry` is your entry widget for the seed
        if seed_value:
            seed = int(seed_value)
            torch.manual_seed(seed) 
        image = pipe(
            prompt.get(),
            guidance_scale=guidance_scale,
            num_inference_steps= int(inference_value_entry.get()),
            height=image_height, 
            width= image_width,
            # seed= int(seed_value_entry.get())  # modify
        )["images"][0]        # height=1024, width=1024,              
        image.save("generated_image.png") 
    original_img = image
    border_width = 5
    frame_width = img_border_frame.winfo_width()- (2 * border_width)
    frame_height = img_border_frame.winfo_height()- (2 * border_width)
    resized_img = original_img.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
    img = ImageTk.PhotoImage(resized_img)          
    img_ref.configure(image=img)
    img_ref.original_img = img #keep a reference to the image
     


    # unconditional image captioning
    inputs = processor(original_img, return_tensors="pt").to("cuda", torch.float16)
    out = blip_model.generate(**inputs, max_length=64)
    unconditional_caption = processor.decode(out[0], skip_special_tokens=True)
    unconditional_caption_label.configure(text="Unconditional: " + unconditional_caption)


    # conditional image captioning
    text = "A photograph of"
    inputs = processor(original_img, text, return_tensors="pt").to("cuda", torch.float16)
    out = blip_model.generate(**inputs, max_length=64)
    conditional_caption = processor.decode(out[0], skip_special_tokens=True)
    conditional_caption_label.configure(text="Conditional: " + conditional_caption)




# prompt box 
prompt = ctk.CTkEntry( app , width = 365 , height = 35, text_font=("Arial", 16), text_color="black",corner_radius= 10, fg_color="white",placeholder_text="Type your prompt here") 
# prompt.place(x=44, y=10) 
prompt.grid(row=0, column=0, columnspan=2, sticky= 'ew', padx=(40,150), pady=(61,10)) # Adjust padding as needed  

# Clear button widget 
clear_button = ctk.CTkButton(app, width = 85, height = 28,  text="Clear", command=clear_prompt,text_font=("Arial", 20), text_color="white", corner_radius= 10, fg_color="blue")
clear_button.grid(row=0,column= 1, padx=(140,40),pady=(60,10), sticky='ew')

# label for the guidance_scale_slider
guidance_scale_label = ctk.CTkLabel(app, text=f"Guidance Scale : {guidance_scale:.2f}", text_color='white',  text_font=("Arial", 18))
guidance_scale_label.grid(row=1, column=0, padx=(0,0), pady=(0,10) ,sticky='ew')  # Adjust padding as needed


#  Create the slider for adjusting guidance_scale
guidance_scale_slider = ctk.CTkSlider(app, from_=0, to=15, height=15, width =120,command=update_guidance_scale)
guidance_scale_slider.grid(row=1, column=0, rowspan=1, padx=(40,0), pady=(35,10),sticky='ew')# guidance_scale_slider.pack()  # Adjust layout as needed

# Option menu for image size 
optionmenu_var = ctk.StringVar(value="256*256")  # default value 
optionmenu = ctk.CTkOptionMenu(app, height=30 ,values=["512*512 ", "512*720", "512*1024", "720*512", "720*720", "720*1024", "1024*512", "1024*720", "1024*1024"],command=optionmenu_callback, width=160,text_font=("Arial", 19), variable=optionmenu_var)
optionmenu.grid(row=1, column=1, padx=(20,40), pady=(0,10), sticky='we')


# Inference Steps Label
inference_steps_label = ctk.CTkLabel(app, text="Inference Steps: ", text_font=("Arial", 18), text_color="white")
inference_steps_label.grid(row=2, column=0, padx=(38,86), pady=0,)

# Inference Value input 
inference_steps = ctk.StringVar(value="10")
inference_value_entry = ctk.CTkEntry(app, width = 68,
    height = 28, text_font=("Arial", 15), text_color="black", fg_color="white", placeholder_text="Enter No. Steps",textvariable = inference_steps )
inference_value_entry.grid(row=2, column=0, padx=(230,27), pady=0, )

#  sticky='w'
# sticky='e'




# row 3 for future use  # add the see functioning here 
# Seed Value Label
seed_label = ctk.CTkLabel(app, text="Seed Value:", text_font=("Arial", 18), text_color="white")
seed_label.grid(row=2, column=1, padx=(0,145), pady=(0), )



# Seed Value input
seed_value = ctk.StringVar(value= None )
seed_value_entry = ctk.CTkEntry(app, width=102, height=29, text_font=("Arial", 15), text_color="black", fg_color="white", placeholder_text="Enter here!", textvariable=seed_value)
seed_value_entry.grid(row=2, column=1, padx=(0,40), pady=(0),sticky='e' )






# Create a CTkFrame with a grey background to act as the border
img_border_frame = ctk.CTkFrame(app,  border_color="grey", border_width=2, corner_radius=13)
img_border_frame.grid(row=4, padx=40, pady=10, columnspan=2, sticky='ew')

# Adjust the img_ref label to be inside the frame, with slightly smaller dimensions to fit within the border
 
img_ref = ctk.CTkLabel(img_border_frame, height= 300 ,text_font=("Arial", 20),text=" Generated Image ",  corner_radius=13)
img_ref.pack(expand=True, fill='both', padx=5, pady=5)  # Center the label inside the frame


# Labels for captions
conditional_caption_label = ctk.CTkLabel(app, text="",text_color='black', fg_color="lightgrey", width=512, height=40,corner_radius= 12, text_font=("Arial", 12))
# conditional_caption_label.place(x=44, y=632-40 + 10 )
conditional_caption_label.grid(row=5, columnspan=2, sticky='nsew', padx=40, pady= 5)

unconditional_caption_label = ctk.CTkLabel(app, text="",text_color= 'black', fg_color="lightgrey", width=512, height=40, corner_radius= 12, text_font=("Arial", 12))
# unconditional_caption_label.place(x=44, y=632+20) # ADD ON 
unconditional_caption_label.grid(row=6, columnspan=2,sticky= 'nsew', padx=40,pady=5)








  # row 7,8,9 for future use








# Button to trigger image generation
trigger = ctk.CTkButton(height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="blue",text="Generate",corner_radius= 10, command=generate) 
# trigger.configure() 
# trigger.place(x=240, y= 60)   # generate button
trigger.grid(row=10, columnspan=2, padx=20, pady=5, sticky='nsew')
app.bind("<Return>", lambda event= None: generate()) # set enter key to the generate

app.bind('<Control-Left>', move_cursor_word_left)
app.bind('<Control-Right>', move_cursor_word_right)
app.columnconfigure(0, weight=1)
app.mainloop()




#Corner_radius , Fg_color