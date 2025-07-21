import argparse
import os
import os.path as osp
import sys
import warnings

import gradio as gr

warnings.filterwarnings('ignore')

# Model
sys.path.insert(
    0, os.path.sep.join(osp.realpath(__file__).split(os.path.sep)[:-2]))
import wan
from wan.configs import WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video

# Global Var
prompt_expander = None
wan_t2v = None
wan_i2v = None
wan_flf2v = None
wan_vace = None


# Button Func
def prompt_enc(prompt, tar_lang):
    global prompt_expander
    prompt_output = prompt_expander(prompt, tar_lang=tar_lang.lower())
    if prompt_output.status == False:
        return prompt
    else:
        return prompt_output.prompt


def load_model(model_name):
    global wan_t2v, wan_i2v, wan_flf2v, wan_vace, args
    if "t2v" in model_name:
        cfg = WAN_CONFIGS[model_name]
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir_t2v,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
        )
    elif "i2v" in model_name:
        # Add I2V model loading logic here
        pass
    elif "flf2v" in model_name:
        # Add FLF2V model loading logic here
        pass
    elif "vace" in model_name:
        # Add VACE model loading logic here
        pass
    return f"Model {model_name} loaded successfully."


def t2v_generation(txt2vid_prompt, resolution, sd_steps, guide_scale,
                   shift_scale, seed, n_prompt):
    global wan_t2v
    # print(f"{txt2vid_prompt},{resolution},{sd_steps},{guide_scale},{shift_scale},{seed},{n_prompt}")

    W = int(resolution.split("*")[0])
    H = int(resolution.split("*")[1])
    video = wan_t2v.generate(
        txt2vid_prompt,
        size=(W, H),
        shift=shift_scale,
        sampling_steps=sd_steps,
        guide_scale=guide_scale,
        n_prompt=n_prompt,
        seed=seed,
        offload_model=True)

    output_path = "generated_video.mp4"
    cache_video(
        tensor=video[None],
        save_file=output_path,
        fps=16,
        nrow=1,
        normalize=True,
        value_range=(-1, 1))

    return output_path


# Interface
def gradio_interface():
    with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="blue")) as demo:
        gr.Markdown("""
                    <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                        Wan2.1
                    </div>
                    <div style="text-align: center; font-size: 16px; font-weight: normal; margin-bottom: 20px;">
                        Wan: Open and Advanced Large-Scale Video Generative Models.
                    </div>
                    """)

        with gr.Row():
            model_selection = gr.Dropdown(
                label="Select Model",
                choices=list(WAN_CONFIGS.keys()),
                value="t2v-14B"
            )
            load_model_button = gr.Button("Load Model")

        status_text = gr.Textbox(label="Status", interactive=False)

        load_model_button.click(
            fn=load_model,
            inputs=[model_selection],
            outputs=[status_text]
        )

        with gr.Tabs():
            with gr.TabItem("Text-to-Video"):
                with gr.Row():
                    with gr.Column():
                        t2v_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the video you want to generate",
                        )
                        t2v_tar_lang = gr.Radio(
                            choices=["ZH", "EN"],
                            label="Target language of prompt enhance",
                            value="ZH")
                        t2v_run_p_button = gr.Button(value="Prompt Enhance")

                        with gr.Accordion("Advanced Options", open=True):
                            t2v_resolution = gr.Dropdown(
                                label='Resolution(Width*Height)',
                                choices=[
                                    '720*1280', '1280*720', '960*960', '1088*832',
                                    '832*1088', '480*832', '832*480', '624*624',
                                    '704*544', '544*704'
                                ],
                                value='720*1280')

                            with gr.Row():
                                t2v_sd_steps = gr.Slider(
                                    label="Diffusion steps",
                                    minimum=1,
                                    maximum=1000,
                                    value=50,
                                    step=1)
                                t2v_guide_scale = gr.Slider(
                                    label="Guide scale",
                                    minimum=0,
                                    maximum=20,
                                    value=5.0,
                                    step=1)
                            with gr.Row():
                                t2v_shift_scale = gr.Slider(
                                    label="Shift scale",
                                    minimum=0,
                                    maximum=10,
                                    value=5.0,
                                    step=1)
                                t2v_seed = gr.Slider(
                                    label="Seed",
                                    minimum=-1,
                                    maximum=2147483647,
                                    step=1,
                                    value=-1)
                            t2v_n_prompt = gr.Textbox(
                                label="Negative Prompt",
                                placeholder="Describe the negative prompt you want to add"
                            )

                        t2v_run_button = gr.Button("Generate Video")

                    with gr.Column():
                        t2v_result = gr.Video(
                            label='Generated Video', interactive=False, height=600)

                t2v_run_p_button.click(
                    fn=prompt_enc,
                    inputs=[t2v_prompt, t2v_tar_lang],
                    outputs=[t2v_prompt])

                t2v_run_button.click(
                    fn=t2v_generation,
                    inputs=[
                        t2v_prompt, t2v_resolution, t2v_sd_steps, t2v_guide_scale, t2v_shift_scale,
                        t2v_seed, t2v_n_prompt
                    ],
                    outputs=[t2v_result],
                )
            with gr.TabItem("Image-to-Video"):
                gr.Markdown("Image-to-Video generation is not yet implemented.")
            with gr.TabItem("First-Last-Frame-to-Video"):
                gr.Markdown("First-Last-Frame-to-Video generation is not yet implemented.")
            with gr.TabItem("VACE"):
                gr.Markdown("VACE is not yet implemented.")
        js_code = """
        function(theme) {
            if (theme) {
                document.body.classList.add('dark');
            } else {
                document.body.classList.remove('dark');
            }
            return theme;
        }
        """
        theme_switcher = gr.Checkbox(label="Dark Mode", value=False)
        theme_switcher.change(lambda x: gr.themes.Default(primary_hue="blue", secondary_hue="blue") if not x else gr.themes.Base(), None, demo)
    return demo


# Main
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt or image using Gradio")
    parser.add_argument(
        "--ckpt_dir_t2v",
        type=str,
        default="cache",
        help="The path to the T2V checkpoint directory.")
    parser.add_argument(
        "--ckpt_dir_i2v",
        type=str,
        default="cache",
        help="The path to the I2V checkpoint directory.")
    parser.add_argument(
        "--ckpt_dir_flf2v",
        type=str,
        default="cache",
        help="The path to the FLF2V checkpoint directory.")
    parser.add_argument(
        "--ckpt_dir_vace",
        type=str,
        default="cache",
        help="The path to the VACE checkpoint directory.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = _parse_args()

    print("Step1: Init prompt_expander...", end='', flush=True)
    if args.prompt_extend_method == "dashscope":
        prompt_expander = DashScopePromptExpander(
            model_name=args.prompt_extend_model, is_vl=False)
    elif args.prompt_extend_method == "local_qwen":
        prompt_expander = QwenPromptExpander(
            model_name=args.prompt_extend_model, is_vl=False, device=0)
    else:
        raise NotImplementedError(
            f"Unsupport prompt_extend_method: {args.prompt_extend_method}")
    print("done", flush=True)

    demo = gradio_interface()
    demo.launch(server_name="0.0.0.0", share=False, server_port=7860)
