def test_replay_and_model_utils_update_paths():
    from rl_puffer_sac_replay_bundle import build_replay_sac_bundle
    from rl_puffer_sac_replay_paths_lib import (
        run_replay_actor_and_q_paths,
        run_replay_buffer_paths,
        run_sac_update_path,
    )

    b = build_replay_sac_bundle()
    run_replay_actor_and_q_paths(b)
    run_replay_buffer_paths(b)
    run_sac_update_path(b)
