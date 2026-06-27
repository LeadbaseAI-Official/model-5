import os
import time
import json
import base64
import re
import subprocess
import uvicorn
import threading
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from github import Github, Auth
from github.GithubException import UnknownObjectException

from model import run_model_query

app: FastAPI = FastAPI(title="Local GGUF LLM API Server")

class ChatRequest(BaseModel):
    prompt: str

# Global handle for cloudflared process
tunnel_process: Optional[subprocess.Popen] = None

# ---------------------------------------------------------------------------
# Cloudflare Tunnel Manager
# ---------------------------------------------------------------------------
def start_cloudflare_tunnel() -> Optional[str]:
    global tunnel_process
    cmd: str = "./cloudflared" if os.path.exists("./cloudflared") else "cloudflared"
    
    try:
        subprocess.run([cmd, "--version"], capture_output=True, check=True)
    except Exception as e:
        print(f"cloudflared binary not found or not working: {e}. Running without tunnel.", flush=True)
        return None

    print(f"Starting cloudflared tunnel using: {cmd}", flush=True)
    try:
        log_file = open("tunnel.log", "w")
        tunnel_process = subprocess.Popen(
            [cmd, "tunnel", "--url", "http://localhost:8000"],
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        
        # Wait up to 15 seconds to extract the trycloudflare.com URL
        url: Optional[str] = None
        for i in range(15):
            time.sleep(1)
            if os.path.exists("tunnel.log"):
                with open("tunnel.log", "r") as f:
                    content: str = f.read()
                    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
                    if match:
                        url = match.group(0)
                        break
        log_file.close()
        
        if url:
            return url
        else:
            print("Failed to extract Cloudflare tunnel URL from tunnel.log.", flush=True)
            return None
    except Exception as ex:
        print(f"Failed to start cloudflared tunnel process: {ex}", flush=True)
        return None

# ---------------------------------------------------------------------------
# GitHub DNS Updater & Dispatcher
# ---------------------------------------------------------------------------
def update_github_dns(pat: str, org: str, public_url: str, repo_name: str) -> None:
    print(f"Connecting to GitHub using PAT to update dynamic DNS registry...", flush=True)
    try:
        auth_obj: Auth.Token = Auth.Token(pat)
        g: Github = Github(auth=auth_obj)
        
        # Target dns repository
        target_repo_name: str = "dns"
        full_repo_path: str = f"{org}/{target_repo_name}"
        
        repo = g.get_repo(full_repo_path)
        
        # Get current config.json contents
        try:
            contents = repo.get_contents("config.json")
            config_bytes: bytes = base64.b64decode(contents.content)
            sha: str = contents.sha
            try:
                config_data: dict = json.loads(config_bytes.decode("utf-8"))
                print("Successfully loaded existing config.json.", flush=True)
            except Exception as parse_err:
                print(f"Warning: config.json content was not valid JSON ({parse_err}). Initializing fresh dictionary.", flush=True)
                config_data = {}
        except UnknownObjectException:
            config_data = {}
            sha = ""
            print("config.json not found in dns repo. Creating a fresh registry.", flush=True)

        # Set key as the current repository name: public_url
        config_data[repo_name] = public_url
        updated_json: str = json.dumps(config_data, indent=2)
        
        if sha:
            repo.update_file(
                path="config.json",
                message=f"Update {repo_name} endpoint tunnel DNS URL [automated]",
                content=updated_json,
                sha=sha
            )
            print(f"config.json updated successfully with key '{repo_name}'.", flush=True)
        else:
            repo.create_file(
                path="config.json",
                message=f"Create tunnel DNS registry config.json with key '{repo_name}' [automated]",
                content=updated_json
            )
            print(f"config.json created successfully with key '{repo_name}'.", flush=True)
            
    except Exception as e:
        print(f"Error updating GitHub DNS file: {e}", flush=True)

def trigger_self_workflow(pat: str, org: str, repo_name: str) -> None:
    print(f"Triggering self workflow dispatch for repository {repo_name}...", flush=True)
    try:
        auth_obj: Auth.Token = Auth.Token(pat)
        g: Github = Github(auth=auth_obj)
        repo = g.get_repo(f"{org}/{repo_name}")
        default_branch: str = repo.default_branch
        
        # Trigger standard workflow.yml on the default branch
        wf = repo.get_workflow("workflow.yml")
        wf.create_dispatch(default_branch)
        print("Self workflow dispatch triggered successfully.", flush=True)
    except Exception as e:
        print(f"Failed to trigger self workflow: {e}", flush=True)

def shutdown_timer(pat: str, org: str, repo_name: str, duration_hours: float) -> None:
    duration_seconds: float = duration_hours * 3600
    print(f"Graceful shutdown timer started: Server will run for {duration_hours} hours ({duration_seconds} seconds).", flush=True)
    
    time.sleep(duration_seconds)
    
    print("Timer expired. Initiating graceful shutdown and restart...", flush=True)
    
    # 1. Trigger next workflow run
    if pat and repo_name != "test":
        trigger_self_workflow(pat, org, repo_name)
    else:
        print("Local mode or GITHUB_PAT missing, skipping self-dispatch trigger.", flush=True)
        
    # 2. Short wait to allow dispatch request to register
    time.sleep(5)
    
    # 3. Kill cloudflared tunnel
    global tunnel_process
    if tunnel_process:
        try:
            tunnel_process.terminate()
            tunnel_process.wait(timeout=5)
            print("cloudflared tunnel terminated.", flush=True)
        except Exception as te:
            print(f"Error terminating cloudflared: {te}", flush=True)
        
    print("Exiting server process gracefully with code 0.", flush=True)
    os._exit(0)

# ---------------------------------------------------------------------------
# Startup Event
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup_event() -> None:
    pat: str = os.getenv("GITHUB_PAT", "")
    org: str = os.getenv("GITHUB_ORG", "LeadbaseAI-Official")

    # Resolve repo name from standard environment variable
    repo_full: str = os.getenv("GITHUB_REPOSITORY", "")
    repo_name: str = repo_full.split("/")[-1] if "/" in repo_full else "test"

    # Start the shutdown timer thread
    duration_str: str = os.getenv("RUN_DURATION_HOURS", "4.0")
    try:
        duration_hours: float = float(duration_str)
    except ValueError:
        duration_hours = 4.0

    t: threading.Thread = threading.Thread(
        target=shutdown_timer,
        args=(pat, org, repo_name, duration_hours),
        daemon=True
    )
    t.start()

    # Start Cloudflare Quick Tunnel
    public_url: Optional[str] = start_cloudflare_tunnel()
    if public_url:
        print(f"==================================================", flush=True)
        print(f"CLOUDFLARE TUNNEL ESTABLISHED SUCCESSFULLY!", flush=True)
        print(f"Public API Address: {public_url}", flush=True)
        print(f"==================================================", flush=True)
        
        # Write back tunnel DNS to config.json
        if pat:
            update_github_dns(pat, org, public_url, repo_name)
        else:
            print("Warning: GITHUB_PAT not configured. Skipping DNS config.json registration.", flush=True)
    else:
        print("Running server without public tunnel.", flush=True)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/v1/chat")
def chat(req: ChatRequest) -> dict:
    if not req.prompt:
        raise HTTPException(status_code=400, detail="Prompt parameter is required.")
    
    response_text: str = run_model_query(req.prompt)
    return {
        "response": response_text,
        "prompt": req.prompt
    }

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
