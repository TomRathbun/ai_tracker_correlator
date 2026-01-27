"""
Evaluation script for trained tracker correlator model.
"""
import torch
import argparse
import json
from pathlib import Path
from tqdm import tqdm

from src.model_v3 import (
    RecurrentGATTrackerV3, build_sparse_edges, frame_to_tensors,
    build_full_input, model_forward, manage_tracks
)
from src.metrics import TrackingMetrics, format_metrics
from src.config import ExperimentConfig
from src.visualize import (
    visualize_attention_weights, visualize_track_predictions,
    plot_confusion_matrix, plot_existence_probabilities
)


def evaluate_model(checkpoint_path: str, data_file: str, device: torch.device,
                   visualize: bool = True, output_dir: str = "evaluation_results"):
    """
    Evaluate a trained model on test data.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_file: Path to data file (JSONL)
        device: Device to use
        visualize: Whether to generate visualizations
        output_dir: Directory to save results
    """
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = RecurrentGATTrackerV3(
        num_sensors=config.model.num_sensors,
        hidden_dim=config.model.hidden_dim,
        state_dim=config.model.state_dim,
        num_heads=config.model.num_heads,
        edge_dim=config.model.edge_dim,
        emb_dim=config.model.emb_dim
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Load data
    frames = []
    with open(data_file, 'r') as f:
        for line in f:
            try:
                frames.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(frames)} frames from {data_file}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Evaluation
    metrics = TrackingMetrics(match_threshold=config.loss.match_gate)
    active_tracks = []
    
    # For visualization
    predictions_history = []
    sample_frames_for_viz = []
    
    with torch.no_grad():
        for frame_idx, frame_data in enumerate(tqdm(frames, desc="Evaluating")):
            meas, meas_sensor_ids = frame_to_tensors(frame_data, device)
            num_meas = meas.shape[0]
            if num_meas == 0:
                continue
            
            # Ground truth
            gt_tracks = frame_data.get('gt_tracks', [])
            gt_states_dev = torch.tensor(
                [[gt['x'], gt['y'], gt['z'], gt['vx'], gt['vy'], gt['vz']] for gt in gt_tracks],
                dtype=torch.float32, device=device
            )
            num_gt = gt_states_dev.shape[0]
            
            # Build input
            full_x, full_sensor_id, hidden_state, num_tracks = build_full_input(
                active_tracks, meas, meas_sensor_ids, config.model.num_sensors, device
            )
            
            N = full_x.shape[0]
            if N == 0:
                continue
            
            node_type = torch.cat([
                torch.ones(num_tracks, dtype=torch.long, device=device),
                torch.zeros(num_meas, dtype=torch.long, device=device)
            ])
            
            edge_index, edge_attr = build_sparse_edges(
                full_x,
                max_dist=config.tracking.max_edge_dist,
                k=config.tracking.k_neighbors
            )
            
            # Forward pass
            out, new_hidden_full, alpha, existence_probs, existence_logits = model_forward(
                model, full_x, node_type, full_sensor_id, edge_index, edge_attr, hidden_state
            )
            
            # Track management
            selected = manage_tracks(
                active_tracks=active_tracks,
                out=out,
                new_hidden_full=new_hidden_full,
                existence_probs=existence_probs,
                existence_logits=existence_logits,
                alpha=alpha,
                edge_index=edge_index,
                num_tracks=num_tracks,
                num_meas=num_meas,
                init_thresh=config.tracking.init_thresh_late,
                coast_thresh=config.tracking.coast_thresh_late,
                suppress_thresh=config.tracking.suppress_thresh_late,
                del_exist=config.tracking.del_exist,
                del_age=config.tracking.del_age,
                track_cap=config.tracking.track_cap
            )
            
            active_tracks = selected
            
            # Prepare predictions
            if len(selected) == 0:
                pred_states = torch.empty((0, config.model.state_dim), device=device)
            else:
                pred_states = torch.stack([tr['state'] for tr in selected])
            
            # Update metrics
            if num_gt > 0:
                metrics.update(pred_states, gt_states_dev)
            
            # Store for visualization (sample every 10 frames)
            if visualize and frame_idx % 10 == 0:
                predictions_history.append(pred_states.clone())
                sample_frames_for_viz.append({
                    'frame_idx': frame_idx,
                    'pred_states': pred_states.clone(),
                    'gt_states': gt_states_dev.clone(),
                    'measurements': meas.clone(),
                    'edge_index': edge_index.clone(),
                    'attention': alpha.clone() if alpha is not None else None,
                    'positions': full_x.clone(),
                    'num_tracks': num_tracks,
                    'existence_probs': existence_probs.clone()
                })
    
    # Compute final metrics
    final_metrics = metrics.compute()
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(format_metrics(final_metrics))
    
    # Save metrics to file
    with open(output_path / "metrics.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nSaved metrics to {output_path / 'metrics.json'}")
    
    # Generate visualizations
    if visualize and sample_frames_for_viz:
        print("\nGenerating visualizations...")
        
        # Sample a few frames for detailed visualization
        for i, sample in enumerate(sample_frames_for_viz[:5]):
            frame_idx = sample['frame_idx']
            
            # Track predictions
            visualize_track_predictions(
                sample['pred_states'],
                sample['gt_states'],
                sample['measurements'],
                save_path=str(output_path / f"predictions_frame_{frame_idx}.png")
            )
            
            # Attention weights
            if sample['attention'] is not None and sample['attention'].numel() > 0:
                visualize_attention_weights(
                    sample['edge_index'],
                    sample['attention'],
                    sample['positions'],
                    sample['num_tracks'],
                    save_path=str(output_path / f"attention_frame_{frame_idx}.png")
                )
            
            # Confusion matrix
            if sample['pred_states'].shape[0] > 0 and sample['gt_states'].shape[0] > 0:
                plot_confusion_matrix(
                    sample['pred_states'],
                    sample['gt_states'],
                    match_threshold=config.loss.match_gate,
                    save_path=str(output_path / f"confusion_frame_{frame_idx}.png")
                )
            
            # Existence probabilities
            plot_existence_probabilities(
                sample['existence_probs'],
                sample['num_tracks'],
                save_path=str(output_path / f"existence_frame_{frame_idx}.png")
            )
        
        print(f"Visualizations saved to {output_path}/")
    
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate tracker correlator model")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='data/sim_realistic_003.jsonl',
                       help='Path to data file')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, default: auto-detect)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualizations')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Evaluate
    metrics = evaluate_model(
        checkpoint_path=args.checkpoint,
        data_file=args.data,
        device=device,
        visualize=not args.no_viz,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
