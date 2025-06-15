from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
import torch
import uvicorn
import torchaudio
import os

from src.main.inference_lm import knn_vc, load_expanded_set  # Ensure you have these imported correctly

app = FastAPI(title="Voice Conversion API",
              description="API for voice conversion with gender and language options")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define enums for valid gender and language options
class Gender(str, Enum):
    male = "male"
    female = "female"

class Language(str, Enum):
    english = "english"
    thai = "thai"

# Ensure temp and output directories exist
os.makedirs("temp", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Load models once at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
knnvc_model = knn_vc(pretrained=True, progress=True, prematched=True, device=device)

# Reference files mapping for different gender and language combinations
REF_WAV_PATHS = {
    "male_english": ["/home/nuos/ASPEAK/src/audio/alar_eng_0_newspkr.mp3"],
    "female_english": ["/home/nuos/ASPEAK/src/audio/normal_016_002_michelle.wav"],
    "male_thai": ["/home/nuos/ASPEAK/src/audio/normal_1.wav"],
    "female_thai": ["/home/nuos/ASPEAK/src/audio/F_normal_1.wav"]
}

# Default reference file as fallback
DEFAULT_REF_WAV_PATH = "/home/nuos/ASPEAK/src/audio/English_normal_converted.wav"


@app.post("/process-audio/")
async def process_audio(
    file: UploadFile = File(...),
    gender: Gender = Form(Gender.male, description="Gender for voice conversion"),
    language: Language = Form(Language.english, description="Language for voice conversion")
):
    try:
        print(f"Processing audio with gender={gender} and language={language}")
        print(f"Selected gender type: {type(gender)}, value: {gender.value}")
        
        # Save the uploaded file
        file_location = f"temp/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Extract features using knnvc_model
        query_seq = knnvc_model.get_features(file_location).to(device)

        # Get the correct reference file based on gender and language
        ref_key = f"{gender.value}_{language.value}"  # Use .value to ensure we get the string value
        print(f"Looking up reference files with key: {ref_key}")
        ref_wav_paths = REF_WAV_PATHS.get(ref_key, [DEFAULT_REF_WAV_PATH])
        
        # Check if reference file exists
        if not all(os.path.exists(path) for path in ref_wav_paths):
            # Fall back to default if the specified reference doesn't exist
            print(f"Warning: Reference file for {ref_key} not found, using default")
            ref_wav_paths = [DEFAULT_REF_WAV_PATH]
            
        print(f"Using reference files: {ref_wav_paths}")
        
        # Get matching set
        matching_set = knnvc_model.get_matching_set(ref_wav_paths).to(device)

        # Expand matching set based on gender and language
        print(f"Expanding matching set for gender={gender.value} and language={language.value}")
        expanded_set = load_expanded_set(gender, language, device)
        matching_set = torch.cat([matching_set.view(-1, matching_set.shape[-1]), expanded_set], dim=0)

        # Convert to float tensors
        query_seq = query_seq.float()
        matching_set = matching_set.float()

        # Process matching set with fuzzy spectral attention
        out_wav = knnvc_model.match_with_fuzzy_spectral_attention(query_seq, matching_set)

        # Save output audio with gender and language in filename
        base_filename = file.filename.rsplit('.', 1)[0]
        output_filename = f"{base_filename}_{gender.value}_{language.value}_processed.wav"
        output_path = f"output/{output_filename}"
        torchaudio.save(output_path, out_wav.unsqueeze(0), 16000)

        # Clean up temporary file
        os.remove(file_location)

        return FileResponse(output_path, media_type='audio/wav', filename=output_filename)

    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error: {str(e)}\n{traceback_str}")
        return JSONResponse(content={"error": str(e), "traceback": traceback_str}, status_code=500)

@app.get("/available-options/")
async def get_available_options():
    """Return available gender and language options"""
    return {
        "genders": [gender.value for gender in Gender],
        "languages": [language.value for language in Language]
    }

# @app.get("/health/")
# async def health_check():
#     """Simple health check endpoint"""
#     return {"status": "healthy", "device": str(device)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
