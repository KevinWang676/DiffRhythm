import torch
import torchaudio
import numpy as np
from einops import rearrange
import os
import time
import sys
import uvicorn
import requests
from pathlib import Path
import uuid
from typing import Optional, Union
import mimetypes

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import all required modules from infer_utils
from infer.infer_utils import (
    get_reference_latent,
    get_lrc_token,
    get_style_prompt,
    prepare_model,
    get_negative_style_prompt,
    decode_audio
)

# Initialize FastAPI app
app = FastAPI(
    title="Lyrics to Music API", 
    description="API for generating music from lyrics and a reference audio",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
models = None
device = None

# Model for request parameters
class GenerateMusicRequest(BaseModel):
    audio_length: int = 95
    steps: int = 32
    cfg_strength: float = 4.0
    lyrics: str = ""  # Added lyrics field to the model

# Cleanup function to remove temporary files
def cleanup_temp_files(file_paths):
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            print(f"Error cleaning up {file_path}: {e}")

# Function to load models
def load_models():
    global device
    
    assert torch.cuda.is_available(), "only available on gpu"
    device = 'cuda'
    
    print("Loading models from infer_utils.prepare_model()...")
    cfm, tokenizer, muq, vae = prepare_model(device)
    
    return cfm, tokenizer, muq, vae

# Modified get_style_prompt function with format conversion if needed
def get_style_prompt_with_format_handling(muq, audio_path):
    try:
        print(f"Getting style prompt from {audio_path}")
        
        # Try the original function first
        try:
            return get_style_prompt(muq, audio_path)
        except Exception as e:
            print(f"Error with original format, trying to convert: {str(e)}")
            
            # If that fails, try to convert the audio to a compatible format
            temp_mp3_path = f"{audio_path}.converted.mp3"
            
            # Load and resample the audio with torchaudio
            try:
                audio, sr = torchaudio.load(audio_path)
                torchaudio.save(temp_mp3_path, audio, sr, format="mp3")
                print(f"Successfully converted audio to {temp_mp3_path}")
                
                # Try again with the converted file
                result = get_style_prompt(muq, temp_mp3_path)
                
                # Clean up temporary file
                if os.path.exists(temp_mp3_path):
                    os.remove(temp_mp3_path)
                    
                return result
            except Exception as conv_error:
                print(f"Error converting audio: {str(conv_error)}")
                raise RuntimeError(f"Could not process audio file {audio_path}. Original error: {str(e)}, Conversion error: {str(conv_error)}")
    
    except Exception as e:
        print(f"Error getting style prompt: {str(e)}")
        raise

# Inference function
def inference_process(cfm_model, vae_model, cond, text, duration, style_prompt, negative_style_prompt, start_time, steps=32, cfg_strength=4.0):
    with torch.inference_mode():
        # The original code uses: generated, * = cfm_model.sample(...)
        # Handle the unpacking properly
        result = cfm_model.sample(
            cond=cond,
            text=text,
            duration=duration,
            style_prompt=style_prompt,
            negative_style_prompt=negative_style_prompt,
            steps=steps,
            cfg_strength=cfg_strength,
            start_time=start_time
        )
        
        # Extract the generated result (first element if it's a tuple)
        if isinstance(result, tuple):
            generated = result[0]
        else:
            generated = result
        
        generated = generated.to(torch.float32)
        latent = generated.transpose(1, 2)  # [b d t]
    
        output = decode_audio(latent, vae_model, chunked=False)
        
        # Rearrange audio batch to a single sequence
        output = rearrange(output, "b d n -> d (b n)")
        # Peak normalize, clip, convert to int16, and save to file
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return output

# Function to get the correct file extension from content type or filename
def get_file_extension(filename, content_type=None):
    if content_type:
        ext = mimetypes.guess_extension(content_type)
        if ext:
            return ext
    
    # If content type doesn't work, try from filename
    _, ext = os.path.splitext(filename)
    if ext:
        return ext
    
    # Default extension if nothing else works
    return ".mp3"

# Music generation process - using lyrics text directly
async def process_music_generation(lyrics_text, ref_audio_path, output_path, params):
    global models, device
    
    if models is None:
        raise ValueError("Models not loaded")
        
    cfm, tokenizer, muq, vae = models
    
    audio_length = params.audio_length
    if audio_length == 95:
        max_frames = 2048
    elif audio_length == 285:
        max_frames = 6144
    else:
        raise ValueError(f"Unsupported audio_length: {audio_length}")
    
    print(f"Processing lyrics text, length: {len(lyrics_text)} characters")
    # Use lyrics text directly
    lrc = lyrics_text
        
    lrc_prompt, start_time = get_lrc_token(lrc, tokenizer, device)
    print("Lyrics processed successfully")
    
    print(f"Processing reference audio from {ref_audio_path}")
    # Get style prompt from reference audio with format handling
    style_prompt = get_style_prompt_with_format_handling(muq, ref_audio_path)
    print("Reference audio processed successfully")
    
    # Get negative style prompt
    negative_style_prompt = get_negative_style_prompt(device)
    
    # Get reference latent
    print(f"Getting reference latent with max_frames={max_frames}")
    latent_prompt = get_reference_latent(device, max_frames)
    
    # Run inference
    print(f"Starting inference with steps={params.steps}, cfg_strength={params.cfg_strength}")
    s_t = time.time()
    generated_song = inference_process(
        cfm_model=cfm, 
        vae_model=vae, 
        cond=latent_prompt, 
        text=lrc_prompt, 
        duration=max_frames, 
        style_prompt=style_prompt,
        negative_style_prompt=negative_style_prompt,
        start_time=start_time,
        steps=params.steps,
        cfg_strength=params.cfg_strength
    )
    e_t = time.time() - s_t
    print(f"Inference completed in {e_t:.2f} seconds")
    
    # Save output
    print(f"Saving output to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, generated_song, sample_rate=44100)
    print(f"Output saved successfully, size: {os.path.getsize(output_path)} bytes")
    
    return {"output_path": output_path, "inference_time": e_t}

# Endpoint to generate music
@app.post("/generate_music/")
async def generate_music(
    background_tasks: BackgroundTasks,
    lyrics_file: Optional[UploadFile] = File(None),
    reference_audio: Optional[UploadFile] = File(None),
    lyrics_url: Optional[str] = Form(None),
    reference_audio_url: Optional[str] = Form(None),
    lyrics_text: Optional[str] = Form(None),
    audio_length: int = Form(95),
    steps: int = Form(32),
    cfg_strength: float = Form(4.0)
):
    """
    Generate music from lyrics and a reference audio
    
    - **lyrics_file**: The lyrics file (LRC format)
    - **reference_audio**: The reference audio file (MP3, WAV, etc.)
    - **lyrics_url**: URL to the lyrics file (alternative to lyrics_file)
    - **reference_audio_url**: URL to the reference audio file (alternative to reference_audio)
    - **lyrics_text**: Direct input of lyrics text in LRC format (alternative to lyrics_file/lyrics_url)
    - **audio_length**: Length of generated song in seconds (default: 95)
    - **steps**: Number of diffusion steps (default: 32)
    - **cfg_strength**: Classifier-free guidance strength (default: 4.0)
    """
    if models is None:
        raise HTTPException(
            status_code=503, 
            detail="Models failed to load during startup. Please check server logs for details."
        )
    
    # Create temp directory for files
    temp_dir = Path("./temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Generate unique filenames
    unique_id = uuid.uuid4().hex
    files_to_cleanup = []
    
    try:
        # Process lyrics - handle all three input methods
        if lyrics_file:
            lyrics_content = await lyrics_file.read()
            print(f"Lyrics file: {lyrics_file.filename}, size: {len(lyrics_content)} bytes")
            lyrics_text = lyrics_content.decode('utf-8')
        elif lyrics_url:
            print(f"Downloading lyrics from URL: {lyrics_url}")
            response = requests.get(lyrics_url)
            response.raise_for_status()
            lyrics_content = response.content
            print(f"Lyrics from URL, size: {len(lyrics_content)} bytes")
            lyrics_text = lyrics_content.decode('utf-8')
        elif lyrics_text:
            print(f"Using provided lyrics text, size: {len(lyrics_text)} characters")
        else:
            raise HTTPException(status_code=400, detail="Either lyrics_file, lyrics_url, or lyrics_text must be provided")
        
        # Process reference audio file - preserve original extension
        ref_audio_path = None
        
        if reference_audio:
            # Get appropriate file extension
            file_ext = get_file_extension(reference_audio.filename, reference_audio.content_type)
            ref_audio_path = temp_dir / f"ref_audio_{unique_id}{file_ext}"
            
            ref_audio_content = await reference_audio.read()
            print(f"Reference audio file: {reference_audio.filename}, size: {len(ref_audio_content)} bytes")
            with open(ref_audio_path, "wb") as buffer:
                buffer.write(ref_audio_content)
            files_to_cleanup.append(str(ref_audio_path))
            
        elif reference_audio_url:
            print(f"Downloading reference audio from URL: {reference_audio_url}")
            response = requests.get(reference_audio_url)
            response.raise_for_status()
            
            # Get file extension from URL or content type
            file_ext = get_file_extension(
                reference_audio_url, 
                response.headers.get('content-type')
            )
            ref_audio_path = temp_dir / f"ref_audio_{unique_id}{file_ext}"
            
            ref_audio_content = response.content
            print(f"Reference audio from URL, size: {len(ref_audio_content)} bytes")
            with open(ref_audio_path, "wb") as buffer:
                buffer.write(ref_audio_content)
            files_to_cleanup.append(str(ref_audio_path))
        else:
            raise HTTPException(status_code=400, detail="Either reference_audio or reference_audio_url must be provided")
            
        if not os.path.exists(ref_audio_path) or not os.path.getsize(ref_audio_path):
            raise HTTPException(status_code=400, detail="Reference audio file is empty or could not be created")
        
        # Output path
        output_path = temp_dir / f"output_{unique_id}.wav"
        
        # Validate and convert parameters
        try:
            audio_length = int(audio_length)
            steps = int(steps)
            cfg_strength = float(cfg_strength)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid parameter value: {str(e)}")
        
        # Validate audio_length
        if audio_length not in [95, 285]:
            raise HTTPException(status_code=400, detail=f"Unsupported audio_length: {audio_length}. Must be 95 or 285.")
        
        # Create params object
        params = GenerateMusicRequest(
            audio_length=audio_length,
            steps=steps,
            cfg_strength=cfg_strength,
            lyrics=lyrics_text  # Pass lyrics directly in the params
        )
        
        print(f"Generation parameters: {params}")
        
        # Process music generation with direct lyrics text
        result = await process_music_generation(
            lyrics_text,
            str(ref_audio_path),
            str(output_path),
            params
        )
        
        # Schedule cleanup of temporary files
        background_tasks.add_task(
            cleanup_temp_files, 
            files_to_cleanup
        )
        
        # Check if output file exists
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Output file was not generated")
        
        print(f"Generation complete. Output file size: {os.path.getsize(output_path)} bytes")
        
        # Return the output file
        return FileResponse(
            path=str(output_path),
            media_type="audio/wav",
            filename=f"generated_music_{unique_id}.wav",
            background=background_tasks.add_task(cleanup_temp_files, [str(output_path)])
        )
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during music generation: {str(e)}\n{error_details}")
        
        # Clean up files in case of error
        if 'files_to_cleanup' in locals() and 'output_path' in locals():
            cleanup_temp_files(files_to_cleanup + [str(output_path)])
        elif 'files_to_cleanup' in locals():
            cleanup_temp_files(files_to_cleanup)
            
        raise HTTPException(status_code=500, detail=f"Error during music generation: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if models is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Models not loaded"}
        )
    return {"status": "healthy", "message": "API is running and models are loaded"}

# Load models on startup
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global models
    print("Loading models...")
    models = load_models()
    print("Models loaded successfully")

# Run the app
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8030, reload=False)
