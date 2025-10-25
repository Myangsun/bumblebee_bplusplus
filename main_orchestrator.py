"""
Main Orchestrator for Bumblebee Classification Pipeline
Allows running individual workflows or complete pipeline sequences

Available Workflows:
1. Pipeline 1: COLLECT & ANALYZE (Steps 1, 2, 3)
   - Collects GBIF data
   - Analyzes dataset distribution
   - Prepares train/val/test splits

2. Pipeline 2: TRAIN BASELINE (Steps 5, 7)
   - Trains baseline model on GBIF only
   - Tests baseline model performance

3. Pipeline 3: GENERATE SYNTHETIC (Step 4)
   - Generates synthetic images using GPT-4o
   - Requires OpenAI API key

4. Pipeline 4: VALIDATE SYNTHETIC
   - Creates expert validation materials
   - Prepares images for entomologist review

5. Complete Workflow
   - Runs all pipelines in sequence
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Pipeline definitions
PIPELINES = {
    "1": {
        "name": "COLLECT & ANALYZE",
        "description": "Collect GBIF data, analyze distribution, prepare splits",
        "script": "pipeline_collect_analyze.py",
        "steps": [1, 2, 3],
        "requires": []
    },
    "2": {
        "name": "TRAIN BASELINE",
        "description": "Train baseline model (GBIF only) and test performance",
        "script": "pipeline_train_baseline.py",
        "steps": [5, 7],
        "requires": ["pipeline_collect_analyze.py"]
    },
    "3": {
        "name": "GENERATE SYNTHETIC",
        "description": "Generate synthetic images using GPT-4o (requires API key)",
        "script": "pipeline_generate_synthetic.py",
        "steps": [4],
        "requires": ["pipeline_collect_analyze.py"]
    },
    "4": {
        "name": "VALIDATE SYNTHETIC",
        "description": "Create expert validation materials for synthetic images",
        "script": "pipeline_validate_synthetic.py",
        "steps": ["V"],
        "requires": ["pipeline_generate_synthetic.py"]
    }
}

# Preset workflow sequences
WORKFLOWS = {
    "basic": {
        "name": "BASIC WORKFLOW",
        "description": "Collect data and train baseline model",
        "pipelines": ["1", "2"]
    },
    "full": {
        "name": "FULL WORKFLOW",
        "description": "Complete pipeline including synthetic augmentation",
        "pipelines": ["1", "2", "3", "4"]
    },
    "synthetic": {
        "name": "SYNTHETIC WORKFLOW",
        "description": "From baseline through synthetic generation and validation",
        "pipelines": ["1", "2", "3", "4"]
    }
}


def check_dependencies(pipeline_id: str) -> Tuple[bool, List[str]]:
    """Check if required dependencies exist for a pipeline"""
    pipeline = PIPELINES[pipeline_id]
    missing = []

    for req in pipeline["requires"]:
        script_path = Path(req)
        # Check if previous output directories exist
        if "collect_analyze" in req:
            if not Path("./GBIF_MA_BUMBLEBEES/prepared").exists():
                missing.append(f"Must run '{PIPELINES['1']['name']}' first")
        elif "synthetic" in req:
            if not Path("./SYNTHETIC_BUMBLEBEES").exists():
                missing.append(f"Must run '{PIPELINES['3']['name']}' first")

    return len(missing) == 0, missing


def print_header():
    """Print main menu header"""
    print("\n" + "="*70)
    print("MASSACHUSETTS BUMBLEBEE CLASSIFICATION PIPELINE")
    print("Rare Species Focus: Bombus terricola & Bombus fervidus")
    print("="*70)


def print_pipeline_menu():
    """Print available pipelines"""
    print("\n" + "="*70)
    print("AVAILABLE PIPELINES")
    print("="*70)

    for pipeline_id, pipeline in sorted(PIPELINES.items()):
        print(f"\n[{pipeline_id}] {pipeline['name']}")
        print(f"    Description: {pipeline['description']}")
        print(f"    Steps: {pipeline['steps']}")


def print_workflow_menu():
    """Print available workflow sequences"""
    print("\n" + "="*70)
    print("PRESET WORKFLOW SEQUENCES")
    print("="*70)

    for workflow_id, workflow in WORKFLOWS.items():
        print(f"\n[{workflow_id}] {workflow['name']}")
        print(f"    Description: {workflow['description']}")
        print(f"    Pipelines: {' → '.join(workflow['pipelines'])}")


def run_pipeline(pipeline_id: str) -> bool:
    """Run a single pipeline"""
    if pipeline_id not in PIPELINES:
        print(f"\n✗ Invalid pipeline ID: {pipeline_id}")
        return False

    pipeline = PIPELINES[pipeline_id]

    # Check dependencies
    deps_ok, missing = check_dependencies(pipeline_id)
    if not deps_ok:
        print(f"\n⚠️  Cannot run '{pipeline['name']}':")
        for msg in missing:
            print(f"   - {msg}")
        return False

    # Run pipeline
    print(f"\n\n{'='*70}")
    print(f"RUNNING PIPELINE {pipeline_id}: {pipeline['name']}")
    print(f"{'='*70}")

    try:
        result = subprocess.run(
            ["python", pipeline["script"]],
            cwd=Path.cwd()
        )
        return result.returncode == 0
    except Exception as e:
        print(f"\n✗ Error running pipeline: {e}")
        return False


def run_workflow(workflow_id: str) -> bool:
    """Run a complete workflow sequence"""
    if workflow_id not in WORKFLOWS:
        print(f"\n✗ Invalid workflow ID: {workflow_id}")
        return False

    workflow = WORKFLOWS[workflow_id]
    pipeline_ids = workflow["pipelines"]

    print(f"\n\n{'='*70}")
    print(f"RUNNING WORKFLOW: {workflow['name']}")
    print(f"{'='*70}")
    print(f"Sequence: {' → '.join([PIPELINES[p]['name'] for p in pipeline_ids])}")

    completed = []
    failed = []

    for i, pipeline_id in enumerate(pipeline_ids, 1):
        print(f"\n\n[{i}/{len(pipeline_ids)}] Running {PIPELINES[pipeline_id]['name']}...")

        if run_pipeline(pipeline_id):
            completed.append(pipeline_id)
            print(f"✓ {PIPELINES[pipeline_id]['name']} completed")
        else:
            failed.append(pipeline_id)
            print(f"✗ {PIPELINES[pipeline_id]['name']} failed")
            print("⚠️  Stopping workflow execution")
            break

    # Summary
    print("\n\n" + "="*70)
    print(f"WORKFLOW SUMMARY: {workflow['name']}")
    print("="*70)
    print(f"\nCompleted ({len(completed)}/{len(pipeline_ids)}):")
    for pid in completed:
        print(f"  ✓ {PIPELINES[pid]['name']}")

    if failed:
        print(f"\nFailed:")
        for pid in failed:
            print(f"  ✗ {PIPELINES[pid]['name']}")

    return len(failed) == 0


def interactive_menu():
    """Run interactive menu"""
    while True:
        print_header()
        print("\nWHAT WOULD YOU LIKE TO DO?")
        print("\n[P] Run a single Pipeline")
        print("[W] Run a preset Workflow")
        print("[L] List all options")
        print("[Q] Quit")

        choice = input("\nEnter choice (P/W/L/Q): ").strip().upper()

        if choice == "Q":
            print("\n✓ Exiting pipeline orchestrator")
            break

        elif choice == "L":
            print_pipeline_menu()
            print_workflow_menu()

        elif choice == "P":
            print_pipeline_menu()
            pipeline_id = input("\nEnter pipeline ID (1/2/3/4): ").strip()
            if pipeline_id in PIPELINES:
                run_pipeline(pipeline_id)
            else:
                print(f"\n✗ Invalid pipeline ID: {pipeline_id}")

        elif choice == "W":
            print_workflow_menu()
            workflow_id = input("\nEnter workflow ID (basic/full/synthetic): ").strip().lower()
            if workflow_id in WORKFLOWS:
                if input("\nStart workflow? (y/n): ").lower() == 'y':
                    run_workflow(workflow_id)
            else:
                print(f"\n✗ Invalid workflow ID: {workflow_id}")

        input("\nPress Enter to continue...")


def main():
    """Main execution"""
    if len(sys.argv) > 1:
        # Command-line arguments
        arg = sys.argv[1].lower()

        if arg == "--help" or arg == "-h":
            print_header()
            print_pipeline_menu()
            print_workflow_menu()
            print("\nUsage:")
            print("  python main_orchestrator.py              # Interactive menu")
            print("  python main_orchestrator.py --pipeline 1 # Run pipeline 1")
            print("  python main_orchestrator.py --workflow full # Run full workflow")
            print("  python main_orchestrator.py --list       # List all options")

        elif arg == "--list":
            print_header()
            print_pipeline_menu()
            print_workflow_menu()

        elif arg == "--pipeline" and len(sys.argv) > 2:
            pipeline_id = sys.argv[2]
            run_pipeline(pipeline_id)

        elif arg == "--workflow" and len(sys.argv) > 2:
            workflow_id = sys.argv[2].lower()
            if input(f"Start {WORKFLOWS.get(workflow_id, {}).get('name', 'Unknown')} workflow? (y/n): ").lower() == 'y':
                run_workflow(workflow_id)

        else:
            print(f"Unknown argument: {arg}")
            print("Use --help for usage information")

    else:
        # Interactive menu
        interactive_menu()


if __name__ == "__main__":
    main()
