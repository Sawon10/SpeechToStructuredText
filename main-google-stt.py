import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from google.cloud import speech
from pydub import AudioSegment

# Load environment variables
load_dotenv()
AUDIO_FOLDER = os.getenv("AUDIO_FOLDER", "sample-speech-files")
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Set the path to the service account key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH

# Define state
class SpeechState(TypedDict):
    audio_paths: List[str]
    transcripts: List[str]

def convert_to_mono(input_path: str, output_path: str):
    sound = AudioSegment.from_file(input_path)
    sound = sound.set_channels(1)
    sound = sound.set_frame_rate(16000)  # Recommended for STT
    sound.export(output_path, format="wav")

# Agent node: speech-to-text
# def speech_to_text(state: SpeechState) -> SpeechState:
#     client = speech.SpeechClient()
#     transcripts = []

#     for path in state["audio_paths"]:
#         print(f"🎧 Transcribing: {path}")
#         with open(path, "rb") as audio_file:
#             content = audio_file.read()

#         audio = speech.RecognitionAudio(content=content)
#         config = speech.RecognitionConfig(
#             encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#             language_code="en-US",
#         )

#         # Convert to mono if necessary
#         mono_path = path.replace(".wav", "-mono.wav")
#         convert_to_mono(path, mono_path)
#         # Then use `mono_path` in your transcription
#         response = client.recognize(config=config, audio=audio)

#         transcript = " ".join([result.alternatives[0].transcript for result in response.results])
#         transcripts.append(transcript)

#     return {**state, "transcripts": transcripts}

def speech_to_text(state: SpeechState) -> SpeechState:
    from google.cloud import speech_v1p1beta1 as speech

    client = speech.SpeechClient()
    audio_paths = state["audio_paths"]
    transcripts = []

    for path in audio_paths:
        print(f"🎧 Transcribing: {path}")
        
        # Convert stereo to mono first
        mono_path = path.replace(".wav", "-mono.wav")
        convert_to_mono(path, mono_path)

        # Read audio content
        with open(mono_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US"
        )

        response = client.recognize(config=config, audio=audio)

        # Combine transcript
        full_transcript = " ".join([result.alternatives[0].transcript for result in response.results])
        transcripts.append(full_transcript)

    return {**state, "transcripts": transcripts}

# LangGraph setup
graph = StateGraph(SpeechState)
graph.add_node("speech_to_text", speech_to_text)
graph.set_entry_point("speech_to_text")
graph.add_edge("speech_to_text", END)
compiled_graph = graph.compile()

# Collect audio file paths from the folder
def get_audio_paths(folder: str) -> List[str]:
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".wav", ".flac"))
    ]

# Run the graph
if __name__ == "__main__":
    audio_paths = get_audio_paths(AUDIO_FOLDER)
    initial_state = {
        "audio_paths": audio_paths,
        "transcripts": []
    }
    final_state = compiled_graph.invoke(initial_state)

    print("\n📝 Transcribed Results:")
    for i, transcript in enumerate(final_state["transcripts"]):
        print(f"\nFile {i+1}: {audio_paths[i]}")
        print(transcript)
