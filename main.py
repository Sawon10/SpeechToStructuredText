import os
import openai
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Folder containing the audio files
AUDIO_FOLDER = "sample-speech-files"

# Define state
class SpeechState(TypedDict):
    audio_paths: List[str]
    transcripts: List[str]

# Agent node: speech-to-text
def speech_to_text(state: SpeechState) -> SpeechState:
    audio_paths = state["audio_paths"]
    transcripts = []

    for path in audio_paths:
        print(f"🎧 Transcribing: {path}")
        with open(path, "rb") as audio_file:
            response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            transcripts.append(response.text)

    return {**state, "transcripts": transcripts}

# LangGraph setup
graph = StateGraph(SpeechState)
graph.add_node("speech_to_text", speech_to_text)
graph.set_entry_point("speech_to_text")
graph.add_edge("speech_to_text", END)
compiled_graph = graph.compile()

# Collect audio file paths from the folder
def get_audio_paths(folder: str) -> List[str]:
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith((".mp3", ".wav", ".m4a"))]

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
