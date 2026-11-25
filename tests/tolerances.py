"""Quantitative acceptance tolerances, declared before the tests were run.

Each entry states where the
number comes from.  If a test fails, the failure is reported; a tolerance is
changed only together with a written reason in docs/model_verification.md.
"""

TOL = dict(
    # --- statics -----------------------------------------------------------
    # Newton solve of the equilibrium converges to round-off; forces ~1e3 N.
    static_weight_rel=1e-9,          # |sum Fn - m g| / (m g)
    equilibrium_accel=1e-8,          # max |d2q/dt2| at the solved equilibrium
    standing_mean_force_rel=1e-4,    # mean sum Fn / W over an unforced 2 s run
    # --- inertia / gravity -------------------------------------------------
    mass_invariance_rel=1e-13,       # M(q) must not depend on g at all
    free_fall_accel_abs=1e-9,        # root accelerates at -g, joints do not move
    # --- structure ---------------------------------------------------------
    deck_freq_table3_hz=1e-4,        # Table 3 prints 4 decimals (+-5e-5 Hz)
    deck_matrix_rel=1e-9,            # same element + lumped mass as OpenSees
    deck_static_rel=1e-8,            # nodal point load: Hermite FE is exact
    mesh_bending_rel=1e-4,           # 40 -> 80 elements, bending modes < 50 Hz
    mesh_torsion_rel=5e-3,           # lumped torsional inertia: ~(n pi/N)^2/24
    modal_truncation_static_rel=2e-2,  # contact-point flexibility, 150 Hz vs all modes
    # --- interface ---------------------------------------------------------
    action_reaction_rel=1e-6,        # virtual-work identity, central differences
    centroidal_rel=1e-9,             # m a_com = sum f + m g
    kernel_rel=1e-10,                # compiled path vs NumPy reference
    cross_solver_rel=1e-9,           # Pinocchio vs MuJoCo (independent engine)
    # --- energy ------------------------------------------------------------
    energy_free_flight_rel=1e-8,     # RK4, dt = 0.25 ms, 1 s of tumbling
    momentum_free_flight_rel=1e-8,
    energy_balance_rel=1e-4,         # |balance drift| / work throughput, impact case
    energy_creation_rel=1e-6,        # passive total energy may not increase
    # --- sensors -----------------------------------------------------------
    sensor_static_abs=1e-10,         # m/s^2
    sensor_rotating_rel=1e-10,
    # --- convergence -------------------------------------------------------
    dt_error_rel=1e-3,               # dt = 0.2 ms vs dt = 0.05 ms, sensor specific force
    dt_min_order=2.0,                # observed order of the fixed-step scheme
    rigid_limit_rel=1e-12,           # deck with zero modes == rigid floor
    # --- body model v2 and walking (declared 2026-11-18, before these tests ran)
    leg_axial_load_rel=1e-2,         # spring force vs weight above the feet (shank a few degrees off vertical)
    apparent_mass_freq_rel=1e-2,     # stored calibration reproduces its frequency target
    apparent_mass_zeta_rel=2e-2,     # ... and its damping target
    apparent_mass_static_rel=2e-2,   # apparent mass at 0.5 Hz equals the body mass
    walk_energy_balance_rel=1e-3,    # work-energy identity while walking, dt = 0.25 ms
                                     # (torque switches at state changes lower the local order)
    walk_centroidal_rel=1e-9,        # m a_com = sum f + m g at every sampled walking state
    # --- 3-D walker (declared 2026-11-19, before this test ran) ---------------
    walk3d_energy_balance_rel=1e-3,  # same integrator and step as the planar walker
    walk3d_centroidal_rel=1e-9,      # all three components (no planar constraint);
                                     # the start-up aid's root force is an external force and is included
)
