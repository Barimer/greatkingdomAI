import sys
import os
import cProfile
import pstats

# Add project path to sys.path
sys.path.append(r"C:\Users\User\source\repos\greatkingdomAI")

from run_validation_20 import play_validation_game

def run_profiling():
    print("Starting profiling for 2 games at Depth 3 sequentially...", flush=True)
    for i in range(1, 3):
        print(f"Starting Game #{i}...", flush=True)
        res = play_validation_game(i)
        print(f"Game #{i} finished. Winner: {res['winner']} | Moves: {res['moves']} | Time: {res['duration_seconds']:.1f}s", flush=True)

if __name__ == "__main__":
    cProfile.run("run_profiling()", "profile.out")
    
    # Print statistics using pstats
    p = pstats.Stats("profile.out")
    p.sort_stats("cumulative")
    print("\n=== TOP 30 CUMULATIVE TIME ===")
    p.print_stats(30)
