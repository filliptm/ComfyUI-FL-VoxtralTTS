MODEL_REPO_ID = "mistralai/Voxtral-4B-TTS-2603"

VOICES = [
    "casual_female",
    "casual_male",
    "cheerful_female",
    "neutral_female",
    "neutral_male",
    "fr_female",
    "fr_male",
    "es_female",
    "es_male",
    "de_female",
    "de_male",
    "it_female",
    "it_male",
    "pt_female",
    "pt_male",
    "nl_female",
    "nl_male",
    "ar_male",
    "hi_female",
    "hi_male",
]

LANGUAGES = [
    "English",
    "French",
    "Spanish",
    "German",
    "Italian",
    "Portuguese",
    "Dutch",
    "Arabic",
    "Hindi",
]

# Device detection
def get_default_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
