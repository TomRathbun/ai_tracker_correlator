import optuna


def objective(trial):
    # Suggested hyperparameters
    init_thresh = trial.suggest_float('init_thresh', 0.3, 0.7)
    coast_thresh = trial.suggest_float('coast_thresh', 0.1, 0.5)
    suppress_thresh = trial.suggest_float('suppress_thresh', 0.2, 0.6)
    fp_mult = trial.suggest_float('fp_mult', 1.0, 5.0)
    del_exist = trial.suggest_float('del_exist', 0.05, 0.2)
    del_age = trial.suggest_int('del_age', 5, 15)
    aux_mult = trial.suggest_float('aux_mult', 1.5, 3.5)
    track_cap = trial.suggest_int('track_cap', 100, 300)

    # Short training loop for fast trials (10 epochs, 60 steps)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RecurrentGATTrackerV3(num_sensors=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    active_tracks = []
    track_history = []
    losses_history = []

    for epoch in range(10):  # Short for HPO
        true_trajectories = []
        for _ in range(num_objects):
            init_pos = torch.randn(3) * 30000.0
            vel = torch.randn(3) * 100.0 + torch.tensor([100.0, 0.0, 0.0])
            true_trajectories.append({'initial_pos': init_pos, 'vel': vel})

        sensor_noises = [20.0 + 30.0 * i for i in range(num_sensors)]

        active_tracks = []  # {'state', 'hidden', 'age', 'logit'}
        epoch_losses = []

        for t in range(60):  # Short steps
        # (full graph build, forward, alpha, suppression with suppress_thresh)
        # (selection with init_thresh/coast_thresh)
        # (deletion with del_exist/del_age)
        # (cap with track_cap)
        # (loss with fp_mult, aux_mult)
        # (backward/step)

        track_history.append(len(active_tracks))
        losses_history.append(np.mean(epoch_losses))  # assume epoch_losses collected

    avg_tracks = np.mean(track_history[-5:])
    avg_loss = np.mean(losses_history[-5:])

    return abs(avg_tracks - 5.0) + avg_loss / 10.0  # Minimize deviation from 5 tracks + low loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)  # 50–100 trials; ~10–30 min on CPU

    print("Best hyperparameters:", study.best_params)
    print("Best objective value:", study.best_value)

    # After HPO, run full training with best params (copy into train_model)