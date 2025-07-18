import gradio as gr
import os
from pathlib import Path
from main_manager import MainManager


# Hardcoded base directory
BASE_DIR = r"C:\Users\nick\OneDrive\Dokumente\Studium\TUM\Master\Semester2\AppliedFoundationModels\work\PathPilot\Data"


def validate_directory(path):
    if os.path.isdir(path):
        return f"✅ Valid directory: {os.path.abspath(path)}"
    else:
        return "❌ Invalid directory path. Please try again."
    
def find_incremental_dirs(base_path):
    results = []
    try:
        for run_dir in os.listdir(base_path):
            run_path = os.path.join(base_path, run_dir)
            if os.path.isdir(run_path) and run_dir.startswith("run_"):
                for sub in os.listdir(run_path):
                    if sub.startswith("incremental_analysis_detailed_"):
                        results.append(os.path.join(BASE_DIR,run_dir, sub))
        return sorted(results)
    except Exception as e:
        return [f"Error: {str(e)}"]

def main():
    manager = MainManager()

    #Structure definition
    with gr.Blocks() as demo:
        gr.Markdown("# 🎬 Custom Video Player")

        with gr.Row():

            # Upload video file
            video_file = gr.File(label="Upload MP4 Video")
            
            with gr.Column():
                dir_choices = find_incremental_dirs(BASE_DIR)
                data_dir = gr.Dropdown(
                    label="Select run + incremental analysis directory",
                    choices=dir_choices,
                    value=dir_choices[0] if dir_choices else None,
                    interactive=True
                )

                submit_btn = gr.Button("Submit")
                feedback = gr.Markdown("")  # Visual feedback placeholder

        


        with gr.Row():

            # Frame display (live preview)
            frame_display = gr.Image(
                value=manager.video_player.get_current_frame,
                every=0.01,
                visible=False  # Initially hidden
            )

            with gr.Column():
                toggle_btn = gr.Button("Play/Pause", visible=False)
                
                explain_scene_btn = gr.Button("Explain Scene", interactive=False, visible=False)

                with gr.Row():
                    spatial_gpt_btn = gr.Button("Chat with Spatial GPT", interactive=False, visible=False)
                    spatial_gpt_speech_state = gr.Textbox(
                        value= manager.speech_recognizer.get_state,
                        label="Speech recognizer state",
                        interactive=False,
                        every=0.1,
                        visible=False
                    )
                spatial_gpt_prompt = gr.Textbox(
                    value="Waiting for your audio input...",
                    label="Prompt",
                    interactive=False,
                    visible=False
                )
                spatial_gpt_output = gr.Textbox(
                    label="AI Answer",
                    interactive=False,
                    visible=False
                )







        #function definition
        video_file.change(fn=manager.load_video, inputs=video_file, outputs=[])
        explain_scene_btn.click(fn=manager.explain_scene)


        # Play and pause controls
        def toggle_manager():
            if manager.is_playing:
                manager.stop()
                return (
                    gr.update(interactive=True),
                    gr.update(interactive=True)
                )
            else:
                manager.start()
                return (
                    gr.update(interactive=False),
                    gr.update(interactive=False)
                )
        toggle_btn.click(fn=toggle_manager,outputs = [spatial_gpt_btn,explain_scene_btn])

            

        # Update function
        def load_data_with_feedback(path):
            success = manager.load_data(path)
            if success:
                return (
                    f"✅ Successfully loaded data from: `{os.path.abspath(path)}`",
                    gr.update(visible=True),  # frame_display
                    gr.update(visible=True),  # toggle_btn
                    gr.update(visible=True),  # explain_scene_btn
                    gr.update(visible=True),  # spatial_gpt_btn
                    gr.update(visible=True),  # spatial_gpt_prompt
                    gr.update(visible=True),  # spatial_gpt_output
                    gr.update(visible=True)   # speech recognizer state
                )
            else:
                return (
                    "❌ Failed to load data. Make sure the directory exists and is valid.",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)   
                )

        # Click binding with all affected outputs
        submit_btn.click(
            fn=load_data_with_feedback,
            inputs=data_dir,
            outputs=[
                feedback,
                frame_display,
                toggle_btn,
                explain_scene_btn,
                spatial_gpt_btn,
                spatial_gpt_prompt,
                spatial_gpt_output,
                spatial_gpt_speech_state
            ]
        )

        #chat with gpt button
        def save_transcription():
            try:
                return manager.record_user_audio_and_transcribe()
            except RuntimeError as e:
                return None
            
        def save_prompt(prompt):
            if prompt is not None:
                return manager.chat_with_spatial_data(prompt)
        
        def save_speak(answer=None):
            if answer is not None:
                manager.audio_player.speak_text(answer)
            
        spatial_gpt_btn.click(fn=save_transcription,outputs=[spatial_gpt_prompt])
        spatial_gpt_prompt.change(fn=save_prompt, inputs=[spatial_gpt_prompt],outputs=[spatial_gpt_output])
        spatial_gpt_output.change(fn=save_speak, inputs=[spatial_gpt_output])


    demo.launch()

if __name__ == "__main__":
    main()
