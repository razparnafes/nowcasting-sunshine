# Setup

##How to setup google collab on our server:

### Starting the Jupyter Notebook Server
1. Connect to machine via ssh
2. Run `./docker.sh` `docker run --gpus all -it --rm -p 8888:8888 -v /data:/workspace/data nvcr.io/nvidia/pytorch:21.03-py3`
3. Run `pip install jupyter_http_over_ws; jupyter serverextension enable --py jupyter_http_over_ws`
4. Start the server by `jupyter notebook   --NotebookApp.allow_origin='https://colab.research.google.com'   --port=8888   --NotebookApp.port_retries=0`

### Connecting to the server
After running the server it should print a URL.
1. Copy the URL
2. Run SSH port forwarding on your machine: `ssh -L 8888:localhost:8888 <server IP> -l nowcasting`
3. Click on the arrow next to "Connect" on the top right corner of the screen, and select "Connect to a local runtime". Then paste the URL (including the token!) - but change "hostname" to "localhost"
4. Enjoy

Our data will be in /workspace/data

