import json

def analyze_results_summary(results_file):
    with open(results_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            for filename, results in data.items():
                # Count occurrences of each accuracy value
                counts = {
                    1: 0,    # Correct answers
                    0: 0,    # "I don't know" responses
                    -1: 0,   # Incorrect answers
                    0.5: 0   # Partially correct answers
                }
                
                # Count each accuracy value
                for result in results:
                    if result == 1:
                        counts[1] += 1
                    elif result == 0:
                        counts[0] += 1
                    elif result == -1:
                        counts[-1] += 1
                    elif result == 0.5:
                        counts[0.5] += 1
                
                # Calculate total
                total = sum(counts.values())
                
                # Print results in the requested format
                print(f"{filename}: 1: {counts[1]}; 0: {counts[0]}, -1: {counts[-1]}, 0.5: {counts[0.5]}, total: {total}")

if __name__ == "__main__":
    analyze_results_summary("results_summary.jsonl") 