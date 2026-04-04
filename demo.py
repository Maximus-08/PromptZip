import os
import time
from dotenv import load_dotenv
load_dotenv()

from server.prompt_zip_environment import PromptZipEnvironment, DATASET
from models import PromptZipAction

def run_demo():
    print("🚀 Initializing PromptZip Environment...")
    env = PromptZipEnvironment()
    
    # Force Mock Mode for demo if no key is present
    if not os.environ.get("GROQ_API_KEY"):
        print("⚠️ No GROQ_API_KEY found, running in deterministic mock mode.")
        env._groq._client = None 
    else:
        print("✅ GROQ_API_KEY found, running LIVE against Llama 3!")

    num_episodes = len(DATASET)
    print(f"\nRunning random demo policy on all {num_episodes} examples...")
    
    total_reward = 0.0
    
    for i in range(num_episodes):
        obs = env.reset(seed=i)
        print(f"\n--- 🏁 EPISODE {i+1}/{num_episodes} START ---")
        print(f"Task: {obs.task_type} | Budget: {obs.token_budget} tokens | Initial length: {obs.token_count} tokens")
        
        span_ids = list(obs.spans.keys())
        # Let's take some simulated actions
        
        episode_reward = 0.0
        
        # Elide the first span if there are enough spans
        if len(span_ids) > 1:
            obs = env.step(PromptZipAction(action_type="elide", span_id=span_ids[0]))
            episode_reward += (obs.reward or 0.0)
            
        # Rephrase the second span if it exists
        if len(span_ids) > 2:
            obs = env.step(PromptZipAction(action_type="rephrase", span_id=span_ids[1]))
            episode_reward += (obs.reward or 0.0)
            
        # Preserve remaining unlocked spans to force all_locked or budget termination
        while not obs.done:
            # Find an unlocked span to preserve; if none, just preserve the first span repeatedly to hit step limit
            unlocked = [s for s in obs.spans.keys() if s not in obs.locked_spans]
            target_span = unlocked[0] if unlocked else list(obs.spans.keys())[0]
            
            obs = env.step(PromptZipAction(action_type="preserve", span_id=target_span))
            episode_reward += (obs.reward or 0.0)
                
        print(f"Episode Done: {obs.done} | Total Episode Reward: {episode_reward:.4f}")
        total_reward += episode_reward
        
        # Small sleep to respect free-tier Groq API rate limits
        time.sleep(1)
        
    print("\n==================================")
    print("🛑 ALL EPISODES FINISHED")
    print(f"Total Combined Reward: {total_reward:.4f}")
    print("==================================")

if __name__ == "__main__":
    run_demo()
