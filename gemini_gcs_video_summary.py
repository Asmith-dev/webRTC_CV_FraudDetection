#!/usr/bin/env python3
"""Simple demo: download a small video from a Google Cloud Storage URL and ask Gemini to summarise it.

Requires:
  pip install google-genai requests

Before running, set the environment variable GOOGLE_API_KEY with your Gemini API key.

Usage:
  python gemini_gcs_video_summary.py <signed_or_public_gcs_url>
"""
import os
import sys
from pathlib import Path
from typing import Optional

import requests
from google import genai

MAX_SIZE_BYTES = 100 * 1024 * 1024  # 40 MB limit imposed by current API
MODEL_NAME = "gemini-2.5-flash"  # supports vision + video

#set GOOGLE_API_KEY=AIzaSyBMLRn_D9u5pYgHfzxAJXbFRBFTHbH6O3w

def download_video(url: str) -> bytes:
    """Download the video from the given URL and return raw bytes."""
    print("0")
    resp = requests.get(url, stream=True, timeout=60)
    print("1")
    resp.raise_for_status()
    print("2")
    data: bytes = b""
    for chunk in resp.iter_content(chunk_size=8192):
        if chunk:  # filter out keep-alive chunks
            data += chunk
            if len(data) > MAX_SIZE_BYTES:
                raise ValueError("Video is larger than 40 MB – switch to chunked processing or down-sample.")

    print("hi")
    return data


def summarise_video(video_bytes: bytes) -> Optional[str]:
    """Send the video to Gemini and return the first candidate summary text."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable not set.")

    # Use the newer Google AI Python SDK v2
    client = genai.Client(api_key=api_key)

    # Create the content with video data
    content = genai.types.Content(
        parts=[
            genai.types.Part(
                inline_data=genai.types.Blob(
                    data=video_bytes,
                    mime_type="video/mp4"
                )
            ),
            # genai.types.Part(text="Please analyze this interview video and provide a summary. analyze the frames carefully. Focus on any suspicious behavior, attention patterns, or notable events like eye movements, facial expressions, body language, etc, tell me whether the candidate is lying or not. tell me whether he is reading from somewhere while answering, check the eye gaze for that.do you think he is cheating?")
            genai.types.Part(text="Do you think the candidate is reading from somewhere while answering? check carefully frame by frame along with the audio of the candidate and check the eye gaze and also the audio of the candidate and see the way the candidate answers for that.do you think the candidate is cheating? and also can you give me the report consisting of the flags if any and the time stamps of the flags, also give me the confidence score for each flag if any and the overall confidece that the person is cheating or not")
        ]
    )

    # Generate content using the correct API
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=content
    )

    # Extract and return the textual answer
    if response.candidates and response.candidates[0].content.parts:
        return response.candidates[0].content.parts[0].text
    return None


def main():
    print("Usage: python gemini_gcs_video_summary.py <gcs_video_url>")
        
    # Your GCS URL
    gcs_url = "https://storage.googleapis.com/prod_metantz/interviews_stitched/67fa16f7585c7ca811601528/stitched_video.mp4?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=candidate-platform-service-acc%40candidate-platform-449913.iam.gserviceaccount.com%2F20250710%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250710T130358Z&X-Goog-Expires=604800&X-Goog-SignedHeaders=host&X-Goog-Signature=8462b264dfaee8b90f292b40ad59a8ed08d56a685557f3cc2433a5a639885cd20f8a4b46aa88d66fe03ca23b9b34d211c9c381cac4db6f9f414e3b00992a205ee65f0aecfe9edf435469b2b673276738ddc037369aeb35f62faef6da9e1f1e805785e543ccc13a98318fd7239e73aba0c5c7f4f9f719c3dca0401ac8f9f18975613054e696b22e0dace533364530161213292fb681e136467be0af43997e20378ca7cf8ef4ac357c8a5b2548885aa72d003264e26aa8df54ea64b8df496a9d06957542bc6e1a110aecea473de1e65b650507bdb19b48deb3b35442613976103f233d078355d31955c5b419bf474aed578d4c770e47514b795506aba54d78c815"
    
    print("Downloading video…")
    try:
        video_data = download_video(gcs_url)
    except Exception as e:
        print(f"Failed to download video: {e}")
        sys.exit(1)

    print(f"Downloaded {len(video_data)/1024/1024:.2f} MB – sending to Gemini…")
    try:
        summary = summarise_video(video_data)
    except Exception as e:
        print(f"Gemini API error: {e}")
        sys.exit(1)

    if summary:
        print("\n=== Video Analysis Summary ===\n" + summary)
    else:
        print("No summary returned.")


if __name__ == "__main__":
    main() 