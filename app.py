# app.py
import os, types, gradio as gr

app_code = os.environ.get("APP_CODE", "")


def run_hidden(code_str):
    module = types.ModuleType("hidden_app")
    exec(code_str, module.__dict__)
    if hasattr(module, "main"):
        return module.main()
    else:
        raise RuntimeError("Secret app must define main()")


if app_code:
    demo = run_hidden(app_code)
    demo.launch(server_name="0.0.0.0", server_port=7860)
else:
    raise RuntimeError("APP_CODE is not set. Please add it in your Space Secrets.")
