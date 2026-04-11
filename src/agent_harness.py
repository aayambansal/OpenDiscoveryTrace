"""
OpenDiscoveryTrace: Agent Harness for Scientific Trajectory Generation
Runs on Lambda Labs 8xV100 cluster (orchestration only; inference via API).
"""

import os
import json
import time
import datetime
import traceback
import subprocess
import asyncio
import aiohttp
import hashlib
from pathlib import Path
from typing import Optional
import argparse

# ── Configuration ───────────────────────────────────────────────────────────
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

MAX_STEPS = 30
TEMPERATURE = 0.0
OUTPUT_DIR = Path("trajectories")
OUTPUT_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """You are an AI scientist conducting rigorous scientific research. You must follow a structured scientific workflow for every task:

**Phase 1 - Literature Review**: Search for relevant prior work, cite specific papers, identify what is known and unknown.
**Phase 2 - Hypothesis Formation**: Based on the literature, formulate a clear, testable hypothesis.
**Phase 3 - Experiment Design**: Design a computational experiment to test your hypothesis. Specify methods, tools, and expected outcomes.
**Phase 4 - Execution**: Execute your experiment by writing and running code, querying databases, or performing calculations.
**Phase 5 - Analysis**: Analyze the results. Do they support or refute your hypothesis? Are there confounding factors?
**Phase 6 - Conclusion**: State your final conclusions with appropriate caveats and confidence levels.

At each step, you MUST:
1. State which phase you are in
2. Explain your reasoning (thought)
3. Take a concrete action (tool call, calculation, search, etc.)
4. Report what you observed from the action

If you encounter an error or unexpected result:
- Explicitly acknowledge it
- Diagnose what went wrong
- Revise your approach and try again

Available tools:
- python_exec: Execute Python code (has access to rdkit, numpy, scipy, pandas, requests, biopython)
- web_search: Search the web for information
- pubmed_search: Search PubMed for scientific papers
- api_call: Call a specific API (PubChem, UniProt, KEGG, Materials Project, ChEMBL, etc.)

Format EVERY response as:
PHASE: [current phase name]
THOUGHT: [your reasoning]
ACTION: [tool_name]
ACTION_INPUT: [input to the tool, as JSON if complex]
---
After receiving the observation, continue to the next step.
When you reach your final conclusion, output:
PHASE: Conclusion
FINAL_CLAIM: [your conclusion]
CONFIDENCE: [0.0-1.0]
"""

# ── Tool Execution ──────────────────────────────────────────────────────────

def execute_python(code: str, timeout: int = 60) -> dict:
    """Execute Python code in a sandboxed subprocess."""
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        )
        return {
            "stdout": result.stdout[:5000],
            "stderr": result.stderr[:2000],
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "TIMEOUT: Code execution exceeded 60 seconds", "returncode": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1}


def execute_web_search(query: str) -> dict:
    """Simulate web search using the model's knowledge (placeholder for real search API)."""
    return {"result": f"[Web search results for: {query}] - The model should use its knowledge to provide relevant information.", "status": "simulated"}


def execute_pubmed_search(query: str) -> dict:
    """Search PubMed via E-utilities."""
    import urllib.request, urllib.parse
    try:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        search_url = f"{base_url}esearch.fcgi?db=pubmed&term={urllib.parse.quote(query)}&retmax=5&retmode=json"
        with urllib.request.urlopen(search_url, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        ids = data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return {"results": [], "count": 0}
        id_str = ",".join(ids)
        fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={id_str}&retmode=xml&rettype=abstract"
        with urllib.request.urlopen(fetch_url, timeout=15) as resp:
            xml_text = resp.read().decode()[:8000]
        return {"results": xml_text, "count": len(ids), "ids": ids}
    except Exception as e:
        return {"error": str(e), "results": [], "count": 0}


def execute_api_call(api_name: str, params: dict) -> dict:
    """Call various scientific APIs."""
    import urllib.request, urllib.parse
    try:
        if api_name == "pubchem":
            cid_or_name = params.get("query", "")
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(cid_or_name)}/property/MolecularFormula,MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,CanonicalSMILES/JSON"
            with urllib.request.urlopen(url, timeout=15) as resp:
                return json.loads(resp.read().decode())
        elif api_name == "uniprot":
            query = params.get("query", "")
            url = f"https://rest.uniprot.org/uniprotkb/search?query={urllib.parse.quote(query)}&format=json&size=3"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                return {"results": str(data)[:5000]}
        else:
            return {"note": f"API '{api_name}' call simulated. Params: {params}"}
    except Exception as e:
        return {"error": str(e)}


def dispatch_tool(tool_name: str, tool_input: str) -> dict:
    """Route tool calls to the appropriate handler."""
    if tool_name == "python_exec":
        return execute_python(tool_input)
    elif tool_name == "web_search":
        return execute_web_search(tool_input)
    elif tool_name == "pubmed_search":
        return execute_pubmed_search(tool_input)
    elif tool_name == "api_call":
        try:
            params = json.loads(tool_input)
            api_name = params.pop("api", "unknown")
            return execute_api_call(api_name, params)
        except json.JSONDecodeError:
            return execute_api_call("unknown", {"raw": tool_input})
    else:
        return {"error": f"Unknown tool: {tool_name}"}


# ── Parsing ─────────────────────────────────────────────────────────────────

def parse_response(text: str) -> dict:
    """Parse structured response from the model."""
    result = {
        "phase": "unknown",
        "thought": "",
        "action_type": "none",
        "action_input": "",
        "final_claim": None,
        "confidence": None
    }
    
    lines = text.strip().split("\n")
    current_key = None
    current_value = []
    
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith("PHASE:"):
            if current_key and current_value:
                result[current_key] = "\n".join(current_value).strip()
            current_key = "phase"
            current_value = [line_stripped[6:].strip()]
        elif line_stripped.startswith("THOUGHT:"):
            if current_key and current_value:
                result[current_key] = "\n".join(current_value).strip()
            current_key = "thought"
            current_value = [line_stripped[8:].strip()]
        elif line_stripped.startswith("ACTION:"):
            if current_key and current_value:
                result[current_key] = "\n".join(current_value).strip()
            current_key = "action_type"
            current_value = [line_stripped[7:].strip()]
        elif line_stripped.startswith("ACTION_INPUT:"):
            if current_key and current_value:
                result[current_key] = "\n".join(current_value).strip()
            current_key = "action_input"
            current_value = [line_stripped[13:].strip()]
        elif line_stripped.startswith("FINAL_CLAIM:"):
            if current_key and current_value:
                result[current_key] = "\n".join(current_value).strip()
            current_key = "final_claim"
            current_value = [line_stripped[12:].strip()]
        elif line_stripped.startswith("CONFIDENCE:"):
            if current_key and current_value:
                result[current_key] = "\n".join(current_value).strip()
            try:
                result["confidence"] = float(line_stripped[11:].strip())
            except ValueError:
                result["confidence"] = 0.5
            current_key = None
            current_value = []
        elif line_stripped != "---":
            current_value.append(line)
    
    if current_key and current_value:
        result[current_key] = "\n".join(current_value).strip()
    
    # Fallback: if no FINAL_CLAIM found but text contains conclusion-like patterns
    if result["final_claim"] is None:
        lower_text = text.lower()
        for marker in ["conclusion:", "final answer:", "in conclusion,", "my final claim:", "summary:"]:
            idx = lower_text.find(marker)
            if idx != -1 and ("phase" in lower_text[:idx+100] and ("conclusion" in lower_text[:idx+100] or "6" in lower_text[:idx+100])):
                result["final_claim"] = text[idx + len(marker):].strip()[:500]
                if result["confidence"] is None:
                    result["confidence"] = 0.7
                break
    
    return result


# ── Model API Calls ─────────────────────────────────────────────────────────

async def call_openai(messages: list, session: aiohttp.ClientSession, model_id: str = "gpt-5.4") -> str:
    """Call OpenAI API."""
    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip, deflate"
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_completion_tokens": 4096
    }
    async with session.post("https://api.openai.com/v1/chat/completions", 
                           json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=180)) as resp:
        data = await resp.json(content_type=None)
        if "error" in data:
            raise Exception(f"OpenAI API error: {data['error']}")
        return data["choices"][0]["message"]["content"]


async def call_anthropic(messages: list, session: aiohttp.ClientSession, model_id: str = "claude-opus-4-6") -> str:
    """Call Anthropic API."""
    headers = {
        "x-api-key": ANTHROPIC_KEY,
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip, deflate",
        "anthropic-version": "2023-06-01"
    }
    # Convert from OpenAI format to Anthropic format
    system_msg = ""
    anthropic_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_msg = msg["content"]
        else:
            anthropic_messages.append({"role": msg["role"], "content": msg["content"]})
    
    payload = {
        "model": model_id,
        "max_tokens": 4096,
        "temperature": TEMPERATURE,
        "system": system_msg,
        "messages": anthropic_messages
    }
    async with session.post("https://api.anthropic.com/v1/messages",
                           json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=180)) as resp:
        data = await resp.json(content_type=None)
        if "error" in data:
            raise Exception(f"Anthropic API error: {data['error']}")
        return data["content"][0]["text"]


async def call_gemini(messages: list, session: aiohttp.ClientSession, model_id: str = "gemini-3.1-pro-preview") -> str:
    """Call Gemini API."""
    # Convert to Gemini format
    system_msg = ""
    gemini_contents = []
    for msg in messages:
        if msg["role"] == "system":
            system_msg = msg["content"]
        elif msg["role"] == "user":
            gemini_contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
        elif msg["role"] == "assistant":
            gemini_contents.append({"role": "model", "parts": [{"text": msg["content"]}]})
    
    payload = {
        "contents": gemini_contents,
        "systemInstruction": {"parts": [{"text": system_msg}]},
        "generationConfig": {
            "temperature": TEMPERATURE,
            "maxOutputTokens": 4096
        }
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={GEMINI_KEY}"
    async with session.post(url, json=payload, headers={"Accept-Encoding": "gzip, deflate"},
                           timeout=aiohttp.ClientTimeout(total=180)) as resp:
        data = await resp.json(content_type=None)
        if "error" in data:
            raise Exception(f"Gemini API error: {data['error']}")
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise Exception(f"Gemini unexpected response: {json.dumps(data)[:500]}")


# Model registry: name -> (call_function, api_model_id)
MODEL_REGISTRY = {
    # OpenAI frontier
    "gpt-5.4":          (call_openai,    "gpt-5.4"),
    "gpt-5.4-mini":     (call_openai,    "gpt-5.4-mini"),
    # Anthropic frontier
    "claude-opus-4.6":  (call_anthropic, "claude-opus-4-6"),
    "claude-sonnet-4.6":(call_anthropic, "claude-sonnet-4-6"),
    # Google frontier
    "gemini-3.1-pro":   (call_gemini,    "gemini-3.1-pro-preview"),
    "gemini-3-flash":   (call_gemini,    "gemini-3-flash-preview"),
    "gemini-2.5-pro":   (call_gemini,    "gemini-2.5-pro"),
    "gemini-2.5-flash": (call_gemini,    "gemini-2.5-flash"),
}

# Default set of models to run
DEFAULT_MODELS = ["gpt-5.4", "claude-opus-4.6", "gemini-3.1-pro"]


# ── Main Trajectory Generation ──────────────────────────────────────────────

async def generate_trajectory(task: dict, model_name: str, session: aiohttp.ClientSession) -> dict:
    """Generate a single scientific trajectory for a task using a specific model."""
    
    task_id = task["task_id"]
    trajectory_id = f"{task_id}_{model_name}"
    
    print(f"  [{trajectory_id}] Starting...")
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"SCIENTIFIC TASK:\n{task['prompt']}\n\nDomain: {task['domain']}\nBegin your structured scientific investigation."}
    ]
    
    steps = []
    start_time = time.time()
    total_tokens_est = 0
    total_tool_calls = 0
    total_failures = 0
    total_revisions = 0
    final_claim = None
    final_confidence = None
    success = None
    
    call_fn_base, api_model_id = MODEL_REGISTRY[model_name]
    # Create a partial that binds the model_id
    import functools
    call_fn = functools.partial(call_fn_base, model_id=api_model_id)
    
    for step_idx in range(MAX_STEPS):
        step_start = time.time()
        step_record = {
            "step_id": step_idx,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "phase": "unknown",
            "thought": "",
            "action": {"type": "none", "tool": "", "input": "", "output": ""},
            "observation": "",
            "error": {"occurred": False, "type": None, "message": None},
            "revision_trigger": None,
            "confidence": None,
            "raw_response": ""
        }
        
        try:
            response_text = await call_fn(messages, session)
            step_record["raw_response"] = response_text[:3000]
            total_tokens_est += len(response_text.split()) * 2  # rough estimate
            
            parsed = parse_response(response_text)
            step_record["phase"] = parsed["phase"]
            step_record["thought"] = parsed["thought"]
            
            # Check for final claim
            if parsed["final_claim"]:
                final_claim = parsed["final_claim"]
                final_confidence = parsed["confidence"]
                step_record["action"]["type"] = "conclude"
                steps.append(step_record)
                messages.append({"role": "assistant", "content": response_text})
                break
            
            # Execute tool if requested
            if parsed["action_type"] and parsed["action_type"] != "none":
                tool_name = parsed["action_type"].strip().lower()
                tool_input = parsed["action_input"]
                total_tool_calls += 1
                
                tool_result = dispatch_tool(tool_name, tool_input)
                tool_result_str = json.dumps(tool_result)[:3000]
                
                step_record["action"] = {
                    "type": "tool_call",
                    "tool": tool_name,
                    "input": tool_input[:1000],
                    "output": tool_result_str[:2000]
                }
                step_record["observation"] = tool_result_str[:2000]
                
                # Check for tool errors
                if "error" in tool_result or (isinstance(tool_result, dict) and tool_result.get("returncode", 0) != 0):
                    step_record["error"] = {
                        "occurred": True,
                        "type": "tool_error",
                        "message": tool_result.get("error", tool_result.get("stderr", "Unknown error"))[:500]
                    }
                    total_failures += 1
                
                messages.append({"role": "assistant", "content": response_text})
                if step_idx >= MAX_STEPS - 5:
                    messages.append({"role": "user", "content": f"OBSERVATION:\n{tool_result_str[:3000]}\n\nYou have {MAX_STEPS - step_idx - 1} steps remaining. Please wrap up and provide your FINAL_CLAIM with CONFIDENCE score."})
                elif step_idx >= MAX_STEPS // 2:
                    messages.append({"role": "user", "content": f"OBSERVATION:\n{tool_result_str[:3000]}\n\nContinue. If you have sufficient evidence, proceed to your conclusion with FINAL_CLAIM and CONFIDENCE."})
                else:
                    messages.append({"role": "user", "content": f"OBSERVATION:\n{tool_result_str[:3000]}\n\nContinue your investigation."})
            else:
                # No tool call, just reasoning
                step_record["action"]["type"] = "reasoning"
                messages.append({"role": "assistant", "content": response_text})
                if step_idx >= MAX_STEPS - 5:
                    messages.append({"role": "user", "content": f"You have {MAX_STEPS - step_idx - 1} steps remaining. Please wrap up your investigation and provide your FINAL_CLAIM with CONFIDENCE score."})
                elif step_idx >= MAX_STEPS // 2:
                    messages.append({"role": "user", "content": "Continue your investigation. If you have enough evidence, proceed to your conclusion with FINAL_CLAIM and CONFIDENCE."})
                else:
                    messages.append({"role": "user", "content": "Continue your investigation. Remember to use tools (python_exec, web_search, pubmed_search, api_call) to gather evidence."})
            
            # Check for revision
            if any(kw in response_text.lower() for kw in ["revise", "reconsider", "try a different", "that didn't work", "error", "mistake", "let me correct"]):
                step_record["revision_trigger"] = "self_detected_error"
                total_revisions += 1
            
        except Exception as e:
            step_record["error"] = {
                "occurred": True,
                "type": "api_error",
                "message": str(e)[:500]
            }
            total_failures += 1
            # Add error to messages and let model try to recover
            messages.append({"role": "user", "content": f"An error occurred: {str(e)[:200]}. Please continue your investigation."})
        
        steps.append(step_record)
        step_record["wall_time"] = time.time() - step_start
    
    wall_time = time.time() - start_time
    
    # Determine success
    if final_claim and task.get("ground_truth"):
        # Simple heuristic: check if key terms from ground truth appear in claim
        gt_terms = set(task["ground_truth"].lower().split())
        claim_terms = set(final_claim.lower().split())
        overlap = len(gt_terms & claim_terms) / max(len(gt_terms), 1)
        success = overlap > 0.3  # crude heuristic; will be refined by LLM judge
    elif final_claim:
        success = None  # needs LLM judge
    else:
        success = False  # didn't reach conclusion
    
    trajectory = {
        "trajectory_id": trajectory_id,
        "task_id": task_id,
        "domain": task["domain"],
        "difficulty": task["difficulty"],
        "prompt": task["prompt"],
        "ground_truth": task.get("ground_truth"),
        "model": model_name,
        "trajectory": steps,
        "outcome": {
            "success": success,
            "final_claim": final_claim,
            "confidence": final_confidence,
            "verification": {"method": "pending", "result": None, "score": None},
            "failure_type": None,
            "recovery_attempted": total_revisions > 0,
            "recovery_successful": success if total_revisions > 0 else None
        },
        "metadata": {
            "total_steps": len(steps),
            "total_tokens_est": total_tokens_est,
            "total_tool_calls": total_tool_calls,
            "total_failures": total_failures,
            "total_revisions": total_revisions,
            "wall_time_seconds": round(wall_time, 2),
            "max_steps_reached": len(steps) >= MAX_STEPS,
            "model_version": model_name,
            "temperature": TEMPERATURE,
            "collection_timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }
    }
    
    # Save immediately
    outpath = OUTPUT_DIR / f"{trajectory_id}.json"
    with open(outpath, "w") as f:
        json.dump(trajectory, f, indent=2, default=str)
    
    status = "SUCCESS" if success else ("PENDING" if success is None else "FAILED")
    print(f"  [{trajectory_id}] Done in {wall_time:.1f}s | {len(steps)} steps | {total_tool_calls} tools | {total_failures} errors | {status}")
    
    return trajectory


async def run_model_batch(tasks: list, model_name: str, concurrency: int = 4):
    """Run all tasks for a single model with limited concurrency."""
    semaphore = asyncio.Semaphore(concurrency)
    
    async def bounded_generate(task, session):
        async with semaphore:
            try:
                return await generate_trajectory(task, model_name, session)
            except Exception as e:
                print(f"  FATAL ERROR on {task['task_id']}_{model_name}: {e}")
                traceback.print_exc()
                return None
    
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        print(f"\n{'='*60}")
        print(f"Running {len(tasks)} tasks for model: {model_name}")
        print(f"Concurrency: {concurrency}")
        print(f"{'='*60}\n")
        
        coros = [bounded_generate(task, session) for task in tasks]
        results = await asyncio.gather(*coros)
        
        successful = sum(1 for r in results if r is not None)
        print(f"\n{model_name}: {successful}/{len(tasks)} trajectories completed\n")
        return [r for r in results if r is not None]


async def main():
    parser = argparse.ArgumentParser(description="OpenDiscoveryTrace: Generate scientific agent trajectories")
    all_model_names = list(MODEL_REGISTRY.keys())
    parser.add_argument("--model", type=str, default="default", choices=all_model_names + ["all", "default"])
    parser.add_argument("--domain", type=str, default="all", choices=["drug_discovery", "materials_science", "genomics", "literature", "all"])
    parser.add_argument("--difficulty", type=str, default="all", choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit number of tasks per model")
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent API calls per model")
    parser.add_argument("--task-bank", type=str, default="task_bank.json")
    args = parser.parse_args()
    
    # Load task bank
    with open(args.task_bank) as f:
        task_data = json.load(f)
    
    tasks = task_data["tasks"]
    
    # Filter
    if args.domain != "all":
        tasks = [t for t in tasks if t["domain"] == args.domain]
    if args.difficulty != "all":
        tasks = [t for t in tasks if t["difficulty"] == args.difficulty]
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    
    # Skip already completed
    existing = set(p.stem for p in OUTPUT_DIR.glob("*.json"))
    
    if args.model == "all":
        models = list(MODEL_REGISTRY.keys())
    elif args.model == "default":
        models = DEFAULT_MODELS
    else:
        models = [args.model]
    
    print(f"OpenDiscoveryTrace Generator")
    print(f"Tasks: {len(tasks)} | Models: {models} | Total trajectories: {len(tasks) * len(models)}")
    print(f"Output: {OUTPUT_DIR.absolute()}")
    print(f"Already completed: {len(existing)}")
    
    for model in models:
        # Filter out completed tasks for this model
        model_tasks = [t for t in tasks if f"{t['task_id']}_{model}" not in existing]
        if not model_tasks:
            print(f"\n{model}: All tasks already completed, skipping.")
            continue
        
        print(f"\n{model}: {len(model_tasks)} tasks remaining (skipping {len(tasks) - len(model_tasks)} completed)")
        await run_model_batch(model_tasks, model, concurrency=args.concurrency)
    
    # Summary
    all_trajectories = list(OUTPUT_DIR.glob("*.json"))
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"Total trajectories: {len(all_trajectories)}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
