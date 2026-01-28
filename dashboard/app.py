"""
AI Tracker Research Dashboard

Interactive Streamlit dashboard for experiment management, model visualization,
and feasibility analysis.
"""
import streamlit as st
import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from pathlib import Path
import json
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.mlflow_config import init_mlflow, MLFLOW_EXPERIMENT_NAME
from src.config_schemas import PipelineConfig, DatasetConfig
from src.data_loader import GenericDatasetLoader
from src.augmentor import DataAugmentor


from dashboard.training_backend import get_runner

# Page configuration
st.set_page_config(
    page_title="AI Tracker Research Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize MLflow
# init_mlflow() # Handled by TrainingRunner

# Initialize training runner
runner = get_runner()

# Session state for tracking runs
if 'active_runs' not in st.session_state:
    st.session_state['active_runs'] = []
if 'last_run_id' not in st.session_state:
    st.session_state['last_run_id'] = None

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🎯 AI Tracker Research Dashboard</div>', unsafe_allow_html=True)
st.markdown("**Modular Multi-Sensor Fusion Experimentation Platform**")
st.markdown("---")

# Sidebar - Experiment Control
with st.sidebar:
    st.header("⚙️ Experiment Control")
    
    # Dataset selection
    st.subheader("Dataset")
    dataset_path = st.selectbox(
        "Select Dataset",
        ["data/sim_hetero_001.jsonl", "data/sim_realistic_001.jsonl", "data/sim_clean_001.jsonl"]
    )
    
    # Augmentation settings
    st.subheader("Data Augmentation")
    enable_augmentation = st.checkbox("Enable Augmentation", value=False)
    if enable_augmentation:
        ssr_dropout = st.slider("SSR ID Dropout Rate", 0.0, 0.5, 0.15, 0.05)
        noise_std = st.slider("Position Noise (m)", 0.0, 100.0, 10.0, 5.0)
    else:
        ssr_dropout = 0.0
        noise_std = 0.0
    
    st.markdown("---")
    
    # Model configuration
    st.subheader("Pipeline Configuration")
    state_updater_type = st.selectbox(
        "State Updater",
        ["gnn", "kalman", "hybrid"],
        help="GNN: AI/ML only, Kalman: Classical, Hybrid: GNN with Kalman fallback"
    )
    
    min_hits = st.slider("Min Hits (Track Confirmation)", 1, 10, 5)
    max_age = st.slider("Max Age (Track Coasting)", 1, 10, 5)
    association_threshold = st.slider("Association Threshold", 0.0, 1.0, 0.35, 0.05)
    
    st.markdown("---")
    
    # Experiment metadata
    st.subheader("Experiment Metadata")
    experiment_name = st.text_input("Experiment Name", "feasibility_test")
    
    # Tags
    architecture_tag = st.selectbox("Architecture", ["gnn_hybrid", "gnn_only", "kalman_only"])
    dataset_tag = st.selectbox("Dataset Type", ["sim_hetero", "sim_realistic", "sim_clean"])
    
    tags = {
        "architecture": architecture_tag,
        "dataset": dataset_tag,
        "timestamp": datetime.now().isoformat()
    }
    
    st.markdown("---")
    
    st.markdown("---")
    
    # Advanced Training Settings
    st.subheader("🎓 True Training")
    train_mode = st.toggle("Enable Parameter Updates", value=False, help="Update model weights (expensive)")
    
    if train_mode:
        st.warning("⚠️ True Training is slow. GPU recommended.")
        num_epochs = st.number_input("Epochs", 1, 100, 10)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            value=1e-4
        )
        batch_size = st.number_input("Batch Size", 1, 32, 1)
    else:
        num_epochs = 1
        learning_rate = 1e-4
        batch_size = 1
    
    st.markdown("---")
    
    st.markdown("---")
    
    # One-click ablations
    st.subheader("🚀 Quick Launch")
    
    # Main training button
    if st.button("▶️ Start Training Run", width="stretch", type="primary"):
        with st.spinner("Starting training run..."):
            try:
                run_id = runner.start_run(
                    dataset_path=dataset_path,
                    state_updater_type=state_updater_type,
                    min_hits=min_hits,
                    max_age=max_age,
                    association_threshold=association_threshold,
                    experiment_name=experiment_name,
                    tags=tags,
                    enable_augmentation=enable_augmentation,
                    ssr_dropout=ssr_dropout,
                    noise_std=noise_std,
                    train_mode=train_mode,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size
                )
                st.session_state['last_run_id'] = run_id
                st.success(f"✅ Run completed! ID: {run_id[:8]}")
                st.balloons()
            except Exception as e:
                st.error(f"Training failed: {e}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("GNN Only", width="stretch"):
            with st.spinner("Running GNN-only experiment..."):
                try:
                    gnn_tags = {**tags, "architecture": "gnn_only"}
                    run_id = runner.start_run(
                        dataset_path=dataset_path,
                        state_updater_type="gnn",
                        min_hits=min_hits,
                        max_age=max_age,
                        association_threshold=association_threshold,
                        experiment_name="gnn_only_ablation",
                        tags=gnn_tags,
                        enable_augmentation=enable_augmentation,
                        ssr_dropout=ssr_dropout,
                        noise_std=noise_std
                    )
                    st.success(f"✅ GNN run completed! ID: {run_id[:8]}")
                except Exception as e:
                    st.error(f"GNN run failed: {e}")
    
    with col2:
        if st.button("Kalman Only", width="stretch"):
            with st.spinner("Running Kalman-only experiment..."):
                try:
                    kalman_tags = {**tags, "architecture": "kalman_only"}
                    run_id = runner.start_run(
                        dataset_path=dataset_path,
                        state_updater_type="kalman",
                        min_hits=min_hits,
                        max_age=max_age,
                        association_threshold=association_threshold,
                        experiment_name="kalman_only_ablation",
                        tags=kalman_tags,
                        enable_augmentation=enable_augmentation,
                        ssr_dropout=ssr_dropout,
                        noise_std=noise_std
                    )
                    st.success(f"✅ Kalman run completed! ID: {run_id[:8]}")
                except Exception as e:
                    st.error(f"Kalman run failed: {e}")
    
    if st.button("🔥 GNN vs Kalman Comparison", width="stretch"):
        with st.spinner("Running comparison (this may take a while)..."):
            try:
                results = runner.run_comparison(
                    dataset_path=dataset_path,
                    min_hits=min_hits,
                    max_age=max_age,
                    association_threshold=association_threshold,
                    tags=tags
                )
                st.success(f"✅ Comparison completed!")
                st.info(f"GNN Run: {results['gnn'][:8]}, Kalman Run: {results['kalman'][:8]}")
            except Exception as e:
                st.error(f"Comparison failed: {e}")

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", 
    "📈 Analysis", 
    "🔍 Components", 
    "🏗️ Architecture", 
    "📄 Reports"
])

# Tab 1: Overview
with tab1:
    st.header("Experiment Overview")
    
    # MLflow runs table
    st.subheader("Recent Runs")
    try:
        runs_df = mlflow.search_runs(
            experiment_names=[MLFLOW_EXPERIMENT_NAME],
            max_results=10,
            order_by=["start_time DESC"]
        )
        
        if not runs_df.empty:
            # Select relevant columns and clean up metric names
            display_cols = ['run_id', 'start_time', 'status', 'tags.architecture', 'tags.dataset']
            
            # Map of desired metrics and their possible aliases in MLflow
            metrics_map = {
                'MOTA': ['metrics.MOTA', 'metrics.mota'],
                'MOTP': ['metrics.MOTP', 'metrics.motp'],
                'Precision': ['metrics.Precision', 'metrics.PRECISION', 'metrics.precision'],
                'Recall': ['metrics.Recall', 'metrics.RECALL', 'metrics.recall'],
                'F1': ['metrics.F1', 'metrics.f1'],
                'FP/frame': ['metrics.FP_per_frame', 'metrics.FP_Rate', 'metrics.fp_rate']
            }
            
            for label, aliases in metrics_map.items():
                for alias in aliases:
                    if alias in runs_df.columns:
                        display_cols.append(alias)
                        break
            
            # Format dataframe for display
            df_display = runs_df[display_cols].copy()
            if 'start_time' in df_display.columns:
                df_display['start_time'] = pd.to_datetime(df_display['start_time']).dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(
                df_display.head(10),
                width="stretch",
                height=400
            )
            
            # Reproduce run button
            selected_run_id = st.selectbox("Select run to reproduce", runs_df['run_id'].tolist())
            if st.button("🔄 Reproduce Selected Run"):
                # Load run config and populate sidebar
                run = mlflow.get_run(selected_run_id)
                st.session_state['reproduced_config'] = run.data.params
                st.success(f"Loaded config from run {selected_run_id[:8]}")
        else:
            st.info("No runs found. Start an experiment to see results here!")
    
    except Exception as e:
        st.error(f"Could not load MLflow runs: {e}")
    
    # Real-time monitoring
    st.subheader("Latest Run Results")
    if st.session_state.get('last_run_id'):
        try:
            run = mlflow.get_run(st.session_state['last_run_id'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                mota = run.data.metrics.get('MOTA', run.data.metrics.get('mota', 0))
                st.metric("MOTA", f"{mota:.3f}")
            with col2:
                precision = run.data.metrics.get('Precision', run.data.metrics.get('precision', 0))
                st.metric("Precision", f"{precision:.3f}")
            with col3:
                recall = run.data.metrics.get('Recall', run.data.metrics.get('recall', 0))
                st.metric("Recall", f"{recall:.3f}")
            with col4:
                id_switches = run.data.metrics.get('ID_Switches', run.data.metrics.get('id_switches', 0))
                st.metric("ID Switches", int(id_switches))
            
            st.success(f"Run ID: {st.session_state['last_run_id'][:8]}")
        except Exception as e:
            st.warning(f"Could not load run results: {e}")
    else:
        st.info("Start a training run to see live metrics here")

# Tab 2: Analysis
with tab2:
    st.header("Comparative Analysis")
    
    try:
        runs_df = mlflow.search_runs(
            experiment_names=[MLFLOW_EXPERIMENT_NAME],
            max_results=50
        )
        
        if not runs_df.empty and 'metrics.MOTA' in runs_df.columns:
            # MOTA comparison
            st.subheader("MOTA Across Runs")
            fig = px.bar(
                runs_df.sort_values('metrics.MOTA', ascending=False).head(10),
                x='run_id',
                y='metrics.MOTA',
                color='tags.architecture',
                title="Multi-Object Tracking Accuracy (MOTA) Comparison"
            )
            st.plotly_chart(fig, width="stretch")
            
            # Precision vs Recall scatter
            if 'metrics.Precision' in runs_df.columns and 'metrics.Recall' in runs_df.columns:
                st.subheader("Precision vs Recall")
                fig2 = px.scatter(
                    runs_df,
                    x='metrics.Recall',
                    y='metrics.Precision',
                    color='tags.architecture',
                    size='metrics.MOTA',
                    hover_data=['run_id'],
                    title="Precision-Recall Trade-off"
                )
                st.plotly_chart(fig2, width="stretch")
        else:
            st.info("No metrics available yet. Run experiments to see analysis!")
    
    except Exception as e:
        st.error(f"Analysis error: {e}")

# Tab 3: Component-Specific Views
with tab3:
    st.header("Component Debugging")
    
    component = st.selectbox("Select Component", ["Clutter Filter", "GNN Tracker", "Pairwise Classifiers"])
    
    if component == "Clutter Filter":
        st.subheader("🎯 Clutter Filter Performance")
        st.info("Confusion matrix and ROC curves will appear here after training")
        
        # Placeholder for confusion matrix
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Precision", "99.9%", "+0.2%")
        with col2:
            st.metric("Recall", "92.6%", "+1.5%")
    
    elif component == "GNN Tracker":
        st.subheader("🧠 GNN Attention Analysis")
        st.info("Attention heatmaps showing PSR/SSR feature fusion will appear here")
    
    else:
        st.subheader("🔗 Pairwise Association")
        st.info("PSR-PSR and SSR-ANY classifier metrics will appear here")

# Tab 4: Architecture Visualization
with tab4:
    st.header("Model Architecture Visualization")
    
    model_choice = st.selectbox("Select Model", ["GNN Tracker", "Clutter Classifier", "Pairwise Classifier"])
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate DOT Graph"):
            try:
                from src.model_v3 import RecurrentGATTrackerV3
                from torchviz import make_dot
                
                # Mock input for visualization
                model = RecurrentGATTrackerV3()
                dummy_x = torch.randn(5, 7) # 5 nodes
                dummy_node_type = torch.zeros(5, dtype=torch.long)
                dummy_sensor_id = torch.zeros(5, dtype=torch.long)
                dummy_edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 4]], dtype=torch.long)
                dummy_edge_attr = torch.randn(5, 6) # V3 expects 6 edge features [dp, dv]
                
                out, _, _ = model(dummy_x, dummy_node_type, dummy_sensor_id, dummy_edge_index, dummy_edge_attr)
                dot = make_dot(out, params=dict(model.named_parameters()))
                dot.format = 'png'
                dot_path = Path("dashboard/model_arch")
                dot_path.mkdir(exist_ok=True)
                dot.render(str(dot_path / "model_graph"))
                
                st.image(str(dot_path / "model_graph.png"), caption="Model Computation Graph")
                st.success("Graph generated successfully!")
            except Exception as e:
                st.error(f"Visualization failed: {e}")
    with col2:
        if st.button("Export to ONNX"):
            st.info("ONNX export feature coming soon!")
    
    # Placeholder for Netron viewer
    st.markdown("### Interactive Netron Viewer")
    st.info("ONNX model will be embedded here for interactive exploration")

# Tab 5: Reports
with tab5:
    st.header("Feasibility Reports")
    
    st.subheader("📄 Generate PDF Report")
    
    report_name = st.text_input("Report Name", "feasibility_report")
    include_metrics = st.checkbox("Include Metrics Table", value=True)
    include_graphs = st.checkbox("Include Comparative Graphs", value=True)
    include_notes = st.text_area("Additional Notes", "")
    
    if st.button("Generate Summary Report"):
        if st.session_state.get('last_run_id'):
            try:
                run = mlflow.get_run(st.session_state['last_run_id'])
                report = {
                    "experiment": experiment_name,
                    "run_id": run.info.run_id,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "generated_at": datetime.now().isoformat()
                }
                st.json(report)
                st.download_button(
                    label="Download Report (JSON)",
                    data=json.dumps(report, indent=2),
                    file_name=f"{report_name}.json",
                    mime="application/json"
                )
                st.success("Report generated!")
            except Exception as e:
                st.error(f"Report generation failed: {e}")
        else:
            st.warning("No active run found. Start a run first!")
    
    st.markdown("---")
    
    st.subheader("📤 Upload Custom Dataset")
    uploaded_file = st.file_uploader("Upload JSONL dataset for quick test", type=['jsonl'])
    
    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")
        if st.button("Run Quick Feasibility Test"):
            st.info("Running tracker on uploaded dataset...")

# Footer
st.markdown("---")
st.markdown("**AI Tracker Research Dashboard** | Modular Multi-Sensor Fusion Platform")
