import os
import sys
import json
import uuid
import websocket
import requests
import subprocess

JUPYTER_URL = "https://kkb-production.jupyter-proxy.kaggle.net/k/331131220/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwidHlwIjoiSldUIn0..Yq5GCmuHmIafluaaywnEVw.2g6Pa2BgTG00hngOkIcJjfJ_tQQbBvmuvU0mavP4DICiwz1MumfWD-8riVn5AB_oqbwZykBGo5XKaCLVLTC3aLjgwMNX5G59e7guGeNmXsC0Ma78GgO9bmsqUTJ6FrhyBHen5NANSaTAFzgneb3tf_ee5p0b0xWRcr8HE2jnA8hcDOx56DIqbWE3pWZuhLCIjjmQ0nnoBCjjXoIgpSEY462FJw4x3CRmLvkzo-qOxcQ.zHLrnRZD2HcJSqFf3YtyPw/proxy"
BEST_SCORE_FILE = os.path.join(os.path.dirname(__file__), "best_score.json")

def get_active_kernel_id():
    r = requests.get(f"{JUPYTER_URL}/api/kernels")
    r.raise_for_status()
    kernels = r.json()
    if not kernels:
        raise Exception("No active kernels found on Kaggle!")
    return kernels[0]['id']

def execute_remote_code(code):
    kernel_id = get_active_kernel_id()
    ws_url = JUPYTER_URL.replace("https://", "wss://").replace("http://", "ws://")
    ws_endpoint = f"{ws_url}/api/kernels/{kernel_id}/channels"
    
    ws = websocket.create_connection(ws_endpoint)
    session_id = uuid.uuid4().hex
    msg_id = uuid.uuid4().hex
    
    execute_request = {
        "header": {
            "msg_id": msg_id,
            "username": "username",
            "session": session_id,
            "msg_type": "execute_request",
            "version": "5.3"
        },
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": True,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True
        },
        "buffers": [],
        "parent_header": {}
    }
    
    ws.send(json.dumps(execute_request))
    
    output_lines = []
    try:
        while True:
            response = ws.recv()
            msg = json.loads(response)
            msg_type = msg.get("header", {}).get("msg_type")
            parent_msg_id = msg.get("parent_header", {}).get("msg_id")
            
            if parent_msg_id != msg_id:
                continue
                
            if msg_type == "stream":
                content = msg.get("content", {})
                stream_name = content.get("name")
                text = content.get("text", "")
                if stream_name == "stdout":
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    output_lines.append(text)
                elif stream_name == "stderr":
                    sys.stderr.write(text)
                    sys.stderr.flush()
            elif msg_type == "execute_result":
                data = msg.get("content", {}).get("data", {})
                text_plain = data.get("text/plain", "")
                print(text_plain)
                output_lines.append(text_plain + "\n")
            elif msg_type == "error":
                content = msg.get("content", {})
                ename = content.get("ename", "")
                evalue = content.get("evalue", "")
                traceback = content.get("traceback", [])
                print(f"\nError: {ename}: {evalue}", file=sys.stderr)
                for line in traceback:
                    print(line, file=sys.stderr)
                raise Exception(f"Remote execution failed: {ename}: {evalue}")
            elif msg_type == "status":
                state = msg.get("content", {}).get("execution_state")
                if state == "idle":
                    break
    finally:
        ws.close()
    return "".join(output_lines)

def upload_file(local_path, remote_name):
    print(f"Uploading {os.path.basename(local_path)} to Kaggle...")
    with open(local_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Use python to write the file on Kaggle
    escaped_content = repr(content)
    code = f"with open('{remote_name}', 'w', encoding='utf-8') as f:\n    f.write({escaped_content})\nprint('Uploaded {remote_name} successfully')"
    execute_remote_code(code)

def run_experiment():
    # 1. Upload files
    base_dir = os.path.dirname(__file__)
    upload_file(os.path.join(base_dir, "prepare.py"), "prepare.py")
    upload_file(os.path.join(base_dir, "train.py"), "train.py")
    
    # 2. Run data preparation
    print("\nRunning remote data preparation...")
    execute_remote_code("!python3 prepare.py")
    # 3. Run training
    print("\nRunning remote training...")
    output = execute_remote_code("!export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && torchrun --nproc_per_node=2 train.py")

    
    # 4. Parse results
    val_loss = None
    val_bpt = None
    comp_ratio = None
    
    for line in output.split('\n'):
        if line.startswith("val_loss:"):
            val_loss = float(line.split(":")[1].strip())
        elif line.startswith("val_bpt:"):
            val_bpt = float(line.split(":")[1].strip())
        elif line.startswith("comp_ratio:"):
            comp_ratio = float(line.split(":")[1].strip())
            
    if val_loss is None or val_bpt is None or comp_ratio is None:
        print("Error: Could not parse results from output!", file=sys.stderr)
        return
        
    print(f"\nExperiment Results:")
    print(f"Validation Loss: {val_loss:.6f}")
    print(f"Bits Per Token: {val_bpt:.6f}")
    print(f"Theoretical Compression Ratio: {comp_ratio:.6f}x")
    
    # 5. Check and update best score
    best_loss = None
    if os.path.exists(BEST_SCORE_FILE):
        try:
            with open(BEST_SCORE_FILE, 'r') as f:
                best_score = json.load(f)
                best_loss = best_score.get("val_loss")
        except:
            pass
            
    improved = False
    if best_loss is None or val_loss < best_loss:
        improved = True
        best_loss = val_loss
        with open(BEST_SCORE_FILE, 'w') as f:
            json.dump({"val_loss": val_loss, "val_bpt": val_bpt, "comp_ratio": comp_ratio}, f)
            
    # 6. Log results to TSV and Git ratchet action
    results_tsv = os.path.join(os.path.dirname(__file__), "results.tsv")
    if not os.path.exists(results_tsv):
        with open(results_tsv, 'w', encoding='utf-8') as f:
            f.write("commit\tval_loss\tval_bpt\tcomp_ratio\tstatus\tdescription\n")

    # Get description from CLI argument
    description = sys.argv[1] if len(sys.argv) > 1 else "baseline"

    train_path = "commavq_autoresearch/train.py"
    if improved:
        print(f"\n[NEW BEST] Improved validation loss to {val_loss:.6f}!")
        subprocess.run(["git", "add", train_path], check=True)
        # Check if there are actual changes staged
        diff_res = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if diff_res.returncode != 0:
            commit_msg = f"Improvement: val_loss = {val_loss:.6f}, {description}"
            subprocess.run(["git", "commit", "-m", commit_msg], check=True)
            print("Committed changes to Git.")
        else:
            print("No changes in train.py to commit (baseline run).")
        
        # Get commit hash
        try:
            commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        except:
            commit_hash = "unknown"
            
        with open(results_tsv, 'a', encoding='utf-8') as f:
            f.write(f"{commit_hash}\t{val_loss:.6f}\t{val_bpt:.6f}\t{comp_ratio:.6f}\tkeep\t{description}\n")
    else:
        print(f"\n[NO IMPROVEMENT] val_loss = {val_loss:.6f} (best is {best_loss:.6f}). Reverting train.py...")
        subprocess.run(["git", "checkout", "--", train_path], check=True)
        print("Reverted train.py to last commit.")
        
        with open(results_tsv, 'a', encoding='utf-8') as f:
            f.write(f"discard\t{val_loss:.6f}\t{val_bpt:.6f}\t{comp_ratio:.6f}\tdiscard\t{description}\n")


if __name__ == "__main__":
    run_experiment()
