#!/usr/bin/env python3
"""
Drawer v1 Dataset Pipeline Runner.

This is a multi-stage pipeline that handles the environment split:
- Stage 1 (articulate-anything): SAPIEN rendering  
- Stage 2 (vidbot): MoGe depth + VidBot affordance + lifting + flow

Usage:
    # Run full pipeline
    python drawer_dataset_v1/run_pipeline.py --stage all --max-objects 5
    
    # Run individual stages
    python drawer_dataset_v1/run_pipeline.py --stage render --max-objects 5
    python drawer_dataset_v1/run_pipeline.py --stage process
"""
import os
import sys
import argparse
import subprocess
import json
from pathlib import Path


def run_stage1_render(args):
    """Run SAPIEN rendering in articulate-anything environment."""
    cmd = [
        "conda", "run", "-n", "articulate-anything", "--no-capture-output",
        "python", "drawer_dataset_v1/stage1_render.py",
        "--partnet-root", args.partnet_root,
        "--output", str(args.output),
        "--max-objects", str(args.max_objects),
    ]
    if args.object_ids:
        cmd.extend(["--object-ids"] + args.object_ids)
    
    print(f"\n{'='*60}")
    print("STAGE 1: SAPIEN RENDERING (articulate-anything env)")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=args.project_root)
    return result.returncode == 0


def run_stage2_process(args):
    """Run VidBot processing in vidbot environment."""
    cmd = [
        "conda", "run", "-n", "vidbot", "--no-capture-output",
        "python", "drawer_dataset_v1/stage2_process.py",
        "--input", str(args.output),
        "--device", args.device,
    ]
    
    print(f"\n{'='*60}")
    print("STAGE 2: DEPTH + AFFORDANCE + FLOW (vidbot env)")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=args.project_root)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Drawer v1 Dataset Pipeline")
    parser.add_argument("--stage", choices=["all", "render", "process"], default="all",
                       help="Pipeline stage to run")
    parser.add_argument("--partnet-root", type=str, 
                       default=os.environ.get("PARTNET_MOBILITY_ROOT", "<path-to-partnet-mobility-dataset>"),
                       help="PartNet-Mobility dataset root")
    parser.add_argument("--output", type=str, default="drawer_dataset_v1/output",
                       help="Output directory")
    parser.add_argument("--max-objects", type=int, default=5,
                       help="Maximum objects to process")
    parser.add_argument("--object-ids", nargs="+", default=None,
                       help="Specific object IDs to process")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for processing")
    args = parser.parse_args()
    
    args.project_root = Path(__file__).parent.parent.resolve()
    args.output = args.project_root / args.output
    
    print(f"Project root: {args.project_root}")
    print(f"Output directory: {args.output}")
    
    success = True
    
    if args.stage in ["all", "render"]:
        success = run_stage1_render(args)
        if not success:
            print("Stage 1 (render) failed!")
            return 1
    
    if args.stage in ["all", "process"]:
        success = run_stage2_process(args)
        if not success:
            print("Stage 2 (process) failed!")
            return 1
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
