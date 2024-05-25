#this script starts a jupyter server with all needed libraries loaded 
import modal
import subprocess
import time 
import os 

GPU_CONFIG = modal.gpu.A100(count=1, memory = 80)
MINUTES = 60  # seconds


image_jupy = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "wget")
    .run_commands("pip install git+https://github.com/vgel/repeng.git", gpu = GPU_CONFIG)
    .pip_install("jupyterlab", "colorama")

)

stub = modal.Stub("app_new", image = image_jupy)
JUPYTER_TOKEN = "1234"  # Change me to something non-guessable!


#secrets=[Secret.from_name("mistral-secret")
@stub.function(concurrency_limit=1, timeout= 30000, gpu = GPU_CONFIG)
def run_jupyter(timeout: int):
    jupyter_port = 8888
    with modal.forward(jupyter_port) as tunnel:
        jupyter_process = subprocess.Popen(
            [
                "jupyter",
                "lab",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                f"--port={jupyter_port}",
                "--NotebookApp.allow_origin='*'",
                "--NotebookApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": JUPYTER_TOKEN},
        )

        print(f"Jupyter available at => {tunnel.url}")

        try:
            end_time = time.time() + timeout
            while time.time() < end_time:
                time.sleep(5)
            print(f"Reached end of {timeout} second timeout period. Exiting...")
        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            jupyter_process.kill()


@stub.local_entrypoint()
def main(timeout: int = 10_000):
    # Run the Jupyter Notebook server
    run_jupyter.remote(timeout=timeout)
