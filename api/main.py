from functools import lru_cache
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from .config import Settings
from .utils import process_payload, run_inference, load_library, find_best_matches

app = FastAPI()

@lru_cache()
def get_settings():
    return Settings()


@app.get("/")
def read_root():
    return {"Hello": "World"}


class Recording(BaseModel):
    data:str # Base64 encoded wav audio

@app.post("/")
def search_song(recording: Recording, settings: Settings = Depends(get_settings)):
    encoded_string = recording.data
    input_data = process_payload(encoded_string)
    output = run_inference(input_data)
    library = load_library('library.json')
    matches = find_best_matches(library, output[0])
    return {'matches':matches}
