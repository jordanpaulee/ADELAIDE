import assemblyai as aai
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
import ollama
import constants # import file for API keys

# Defining Adelaide
class adelaide:
    def __init__(self) -> None:
        aai.settings.api_key = constants.assembly_api_key
        self.client = ElevenLabs(
            api_key = constants.elevenlabs_api_key
        )

        self.transcriber = None

        self.full_transcript = [
            {"role": "system", "content":"You are a language model called ADELAIDE and are based on a large language model called Llama 3 developed by Meta, answer the questions being asked in less than 300 characters. Do not bold or asterix anything because this will be passed to a text to speech service."}
        ]


    # Assembly AI Transcription    
    def on_open(self, session_opened: aai.RealtimeSessionOpened):
        #print("Session ID:", session_opened.session_id)
        return


    def on_data(self, transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return

        if isinstance(transcript, aai.RealtimeFinalTranscript):
            print(transcript.text, end="\r\n")
            self.generate_ai_response(transcript)

        else:
            print(transcript.text, end="\r")


    def on_error(self, error: aai.RealtimeError):
        #print("An error occured:", error)
        return
    

    def on_close(self):
        #print("Closing Session")
        return


    def start_transcription(self):
        self.transcriber = aai.RealtimeTranscriber(
            sample_rate=16_000,
            on_data=self.on_data,
            on_error=self.on_error,
            on_open=self.on_open,
            on_close=self.on_close,
        )

        self.transcriber.connect()

        microphone_stream = aai.extras.MicrophoneStream(sample_rate=16_000)
        self.transcriber.stream(microphone_stream)

    def close_transcription(self):
        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None

# Hand-off transcript to Llama 3
    def generate_ai_response(self, transcript):
        self.close_transcription()

        self.full_transcript.append({"role":"user", "content":transcript.text})
        print(f"\nUser:{transcript.text}", end="\r\n")

        ollama_stream = ollama.chat(
            model="llama3",
            messages=self.full_transcript,
            stream=True
        )

        print("Adelaide:", end="\r\n")

        text_buffer = ""
        full_text = ""
        for chunk in ollama_stream:
            text_buffer += chunk['messages']['content']
            if text_buffer.endswith('.'):
                audio_stream = self.client.generate(text=text_buffer,
                                                    model="eleven_turbo_v2",
                                                    stream=True)
                print(text_buffer, end="\n", flush=True)
                stream(audio_stream)
                full_text += text_buffer
                text_buffer = ""

        if text_buffer:
            audio_stream = self.client.generate(text=text_buffer,
                                                    model="eleven_turbo_v2",
                                                    stream=True)
            print(text_buffer, end="\n", flush=True)
            stream(audio_stream)
            full_text += text_buffer
        
        self.full_transcript.append({"role":"assistant", "content":full_text})

        self.start_transcription()

ai_assistant = adelaide()
ai_assistant.start_transcription()