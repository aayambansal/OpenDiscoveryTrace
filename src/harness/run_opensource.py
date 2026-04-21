"""Run open-source model (Qwen2.5-1.5B-Instruct) on 30 tasks."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch, json, time

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)
print("Loaded on GPU!")

tasks = json.load(open("task_bank.json"))["tasks"][:30]
os.makedirs("trajectories_opensource", exist_ok=True)

for i, task in enumerate(tasks):
    msgs = [{"role": "system", "content": "You are an AI scientist. Analyze step by step, then conclude."},
            {"role": "user", "content": task["prompt"]}]
    text_in = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text_in, return_tensors="pt").to("cuda:0")
    start = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    elapsed = time.time() - start
    tid = task["task_id"]
    traj = {
        "trajectory_id": f"{tid}_qwen2.5-1.5b",
        "task_id": tid, "domain": task["domain"], "difficulty": task["difficulty"],
        "model": "Qwen2.5-1.5B-Instruct", "prompt": task["prompt"], "open_source": True,
        "trajectory": [{"step_id": 0, "phase": "single_response", "raw_response": text[:3000],
                        "thought": text[:500], "action": {"type": "reasoning"}, "error": {"occurred": False}}],
        "outcome": {"final_claim": text[-500:], "success": None},
        "metadata": {"total_steps": 1, "total_tool_calls": 0, "total_failures": 0,
                     "total_revisions": 0, "wall_time_seconds": round(elapsed, 2), "model_version": model_id}
    }
    with open(f"trajectories_opensource/{tid}_qwen2.5.json", "w") as f:
        json.dump(traj, f, indent=2)
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/30 done ({elapsed:.1f}s/task)")

print("Done! 30 open-source trajectories generated")
print(f"Files: {len(os.listdir('trajectories_opensource'))}")
