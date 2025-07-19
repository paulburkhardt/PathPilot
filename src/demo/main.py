import gradio as gr
import os
from pathlib import Path
from main_manager import MainManager


BASE_DIR = r"C:\Users\nick\OneDrive\Dokumente\Studium\TUM\Master\Semester2\AppliedFoundationModels\work\PathPilot\Data"
ENABLE_3D_PLOT = False


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
                        results.append(os.path.join(BASE_DIR, run_dir, sub))
        return sorted(results)
    except Exception as e:
        return [f"Error: {str(e)}"]


def main():
    manager = MainManager()

    with gr.Blocks(css="""
    footer {display:none !important}
    #feedback-box.success {
        background-color: #28a745;
        color: white;
        font-weight: bold;
        padding: 8px;
        border-radius: 6px;
        margin-top: 10px;
    }
    #feedback-box.error {
        background-color: #dc3545;
        color: white;
        font-weight: bold;
        padding: 8px;
        border-radius: 6px;
        margin-top: 10px;
    }
    """) as demo:
        gr.Markdown("# 👁️‍🗨️ Spatial AI Assistant for the Visually Impaired\nA multimodal assistant for understanding video scenes and engaging through voice and GPT.")

        with gr.Tabs() as tabs:
            with gr.TabItem("🎬 Video & Data Setup", id=0):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📁 Step 1: Upload a Video File")
                        video_file = gr.File(label="Upload MP4 Video", file_types=[".mp4"])

                    with gr.Column():
                        gr.Markdown("### 📊 Step 2: Select Data Directory")
                        dir_choices = find_incremental_dirs(BASE_DIR)
                        data_dir = gr.Dropdown(
                            label="Run + Incremental Analysis Directory",
                            choices=dir_choices,
                            value=dir_choices[0] if dir_choices else None,
                            interactive=True
                        )
                        submit_btn = gr.Button("🔄 Load Data")

                        feedback = gr.Textbox(
                            label="",
                            interactive=False,
                            show_label=False,
                            max_lines=2,
                            value="",
                            visible=True,
                            container=False,
                            elem_id="feedback-box"
                        )

            with gr.TabItem("🧠 AI Interaction", id=1, visible=False) as ai_tab:
                with gr.Row():
                    with gr.Column():
                        frame_display = gr.Image(
                            value=manager.video_player.get_current_frame,
                            every=0.01,
                            visible=False,
                            label="🎞️ Live Frame Preview"
                        )
                        if ENABLE_3D_PLOT:
                            # NEW: static 3D plot
                            def get_static_plot():
                                if manager.data_manager.data_loaded:
                                    return manager.data_manager.plot_step_static(manager.video_player.step_idx)
                                else:
                                    return None
                            static_plot = gr.Image(
                                value=get_static_plot,
                                every=0.01,
                                visible=False,
                                label="📍 3D Scene (Current Step)"
                            )

                    with gr.Column():
                        toggle_btn = gr.Button("▶️ Play / ⏸️ Pause", visible=False)
                        explain_scene_btn = gr.Button("🖼️ Explain Current Scene", interactive=False, visible=False)

                        gr.Markdown("### 🎤 Talk to the Spatial AI")
                        spatial_gpt_btn = gr.Button("🎙️ Start Voice Input", interactive=False, visible=False)
                        spatial_gpt_speech_state = gr.Textbox(
                            value=manager.speech_recognizer.get_state,
                            label="🎧 Speech Recognition State",
                            interactive=False,
                            every=0.1,
                            visible=False
                        )
                        spatial_gpt_prompt = gr.Textbox(
                            value="Waiting for your audio input...",
                            label="📝 Transcribed Prompt",
                            interactive=False,
                            visible=False
                        )
                        spatial_gpt_output = gr.Textbox(
                            label="💬 GPT Response",
                            interactive=False,
                            visible=False
                        )

        # Bindings
        video_file.change(fn=manager.load_video, inputs=video_file, outputs=[])

        explain_scene_btn.click(fn=manager.explain_scene)

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

        toggle_btn.click(fn=toggle_manager, outputs=[spatial_gpt_btn, explain_scene_btn])

        def load_data_with_feedback(path):
            success = manager.load_data(path)
            num_values = 8
            if ENABLE_3D_PLOT:
                num_values +=1
            updates = []
            for i in range(num_values):
                updates.append(gr.update(visible=success))
            if success:
                return (
                    gr.update(value="✅ Data loaded! Proceed to the AI Interaction tab.", elem_classes="success"),
                    *updates
                )
            else:
                return (
                    gr.update(value="❌ Failed to load data. Please check the selected directory.", elem_classes="error"),
                    *updates
                )

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
                spatial_gpt_speech_state,
                ai_tab,  # tab visibility
                static_plot
            ] if ENABLE_3D_PLOT else
            [
                feedback,
                frame_display,
                toggle_btn,
                explain_scene_btn,
                spatial_gpt_btn,
                spatial_gpt_prompt,
                spatial_gpt_output,
                spatial_gpt_speech_state,
                ai_tab,  # tab visibility
            ]
        )

        # GPT interaction
        def save_transcription():
            try:
                return manager.record_user_audio_and_transcribe()
            except RuntimeError:
                return None

        def save_prompt(prompt):
            if prompt is not None:
                return manager.chat_with_spatial_data(prompt)

        def save_speak(answer=None):
            if answer is not None:
                manager.audio_player.speak_text(answer)

        spatial_gpt_btn.click(fn=save_transcription, outputs=[spatial_gpt_prompt])
        spatial_gpt_prompt.change(fn=save_prompt, inputs=[spatial_gpt_prompt], outputs=[spatial_gpt_output])
        spatial_gpt_output.change(fn=save_speak, inputs=[spatial_gpt_output])

    demo.launch()


if __name__ == "__main__":
    main()
