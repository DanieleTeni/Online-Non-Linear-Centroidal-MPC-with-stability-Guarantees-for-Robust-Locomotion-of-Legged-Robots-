import numpy as np

class FootTrajectoryGenerator:
  def __init__(self, initial, footstep_planner, params):
    self.delta = params['world_time_step']
    self.step_height = params['step_height']
    self.initial = initial
    self.footstep_planner = footstep_planner
    self.plan = self.footstep_planner.plan
    self.first_swing= params['first_swing']
    self.modified_trajectory = None


  def generate_feet_trajectories_at_time(self, time):
    step_index = self.footstep_planner.get_step_index_at_time(time)
    time_in_step = time - self.footstep_planner.get_start_time(step_index)
    phase = self.footstep_planner.get_phase_at_time(time)
    support_foot = self.footstep_planner.plan[step_index]['foot_id']
    swing_foot = 'lfoot' if support_foot == 'rfoot' else 'rfoot'
    single_support_duration = self.footstep_planner.plan[step_index]['ss_duration']

    # if first step, return initial foot poses with zero velocities and accelerations
    if step_index == 0:
        zero_vel = np.zeros(6)
        zero_acc = np.zeros(6)
        return {
            'lfoot': {
                'pos': self.initial['lfoot']['pos'],
                'vel': zero_vel,
                'acc': zero_acc
            },
            'rfoot': {
                'pos': self.initial['rfoot']['pos'],
                'vel': zero_vel,
                'acc': zero_acc
            }
        }

    # if double support, return planned foot poses with zero velocities and accelerations
    if phase == 'ds':
        support_pose = np.hstack((
            self.plan[step_index]['ang'],
            self.plan[step_index]['pos']
        ))
        swing_pose = np.hstack((
            self.plan[step_index + 1]['ang'],
            self.plan[step_index + 1]['pos']
        ))
        zero_vel = np.zeros(6)
        zero_acc = np.zeros(6)
        return {
            support_foot: {
                'pos': support_pose,
                'vel': zero_vel,
                'acc': zero_acc
            },
            swing_foot: {
                'pos': swing_pose,
                'vel': zero_vel,
                'acc': zero_acc
            }
        }
    
    # get positions and angles for cubic interpolation
    start_pos  = self.plan[step_index - 1]['pos']
    target_pos = self.plan[step_index + 1]['pos']
    start_ang  = self.plan[step_index - 1]['ang']
    target_ang = self.plan[step_index + 1]['ang']

    # time variables
    t = time_in_step
    T = single_support_duration

    # cubic polynomial for position and angle
    A = - 2 / T**3
    B =   3 / T**2
    swing_pos     = start_pos + (target_pos - start_pos) * (    A * t**3 +     B * t**2)
    swing_vel     =             (target_pos - start_pos) * (3 * A * t**2 + 2 * B * t   ) / self.delta
    swing_acc     =             (target_pos - start_pos) * (6 * A * t    + 2 * B       ) / self.delta**2
    swing_ang_pos = start_ang + (target_ang - start_ang) * (    A * t**3 +     B * t**2)
    swing_ang_vel =             (target_ang - start_ang) * (3 * A * t**2 + 2 * B * t   ) / self.delta
    swing_ang_acc =             (target_ang - start_ang) * (6 * A * t    + 2 * B       ) / self.delta**2

    # quartic polynomial for vertical position
    A =   16 * self.step_height / T**4
    B = - 32 * self.step_height / T**3
    C =   16 * self.step_height / T**2
    swing_pos[2] =       A * t**4 +     B * t**3 +     C * t**2
    swing_vel[2] = ( 4 * A * t**3 + 3 * B * t**2 + 2 * C * t   ) / self.delta
    swing_acc[2] = (12 * A * t**2 + 6 * B * t    + 2 * C       ) / self.delta**2

    # support foot remains stationary
    support_pos = self.plan[step_index]['pos']
    support_ang = self.plan[step_index]['ang']
    zero_vel = np.zeros(3)
    zero_acc = np.zeros(3)

    # assemble pose, velocity, and acceleration for each foot
    support_data = {
        'pos': np.hstack((support_ang, support_pos)),
        'vel': np.hstack((np.zeros(3), zero_vel)),
        'acc': np.hstack((np.zeros(3), zero_acc))
    }

    swing_data = {
        'pos': np.hstack((swing_ang_pos, swing_pos)),
        'vel': np.hstack((swing_ang_vel, swing_vel)),
        'acc': np.hstack((swing_ang_acc, swing_acc))
    }

    return {
        support_foot: support_data,
        swing_foot: swing_data
    }
  

  def generate_feet_trajectories_pre(self):
    #print("HIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
    support_data_sum=np.empty((0, 1))
    swing_data_sum=np.empty((0, 1))
    l_foot_sum=np.empty((0, 1))
    r_foot_sum=np.empty((0, 1))
    sim_time = int((len(self.plan))/self.delta)
    time=0
    print("preplan")
    print(sim_time)
    for time in range(sim_time):
        #print("I am in")
        step_index = self.footstep_planner.get_step_index_at_time(time)
        time_in_step = time - self.footstep_planner.get_start_time(step_index)
        phase = self.footstep_planner.get_phase_at_time(time)
        support_foot = self.footstep_planner.plan[step_index]['foot_id']
        swing_foot = 'lfoot' if support_foot == 'rfoot' else 'rfoot'
        single_support_duration = self.footstep_planner.plan[step_index]['ss_duration']

        # if first step, return initial foot poses with zero velocities and accelerations
        if step_index == 0:
            zero_vel = np.zeros(6)
            zero_acc = np.zeros(6)
           
            l_foot= {
                    'pos': self.initial['lfoot']['pos'],
                    'vel': zero_vel,
                    'acc': zero_acc
                },
            r_foot= {
                    'pos': self.initial['rfoot']['pos'],
                    'vel': zero_vel,
                    'acc': zero_acc
                }
            l_foot_sum = np.vstack((l_foot_sum,l_foot))
            r_foot_sum   = np.vstack((r_foot_sum,r_foot))
            continue
            

        # if double support, return planned foot poses with zero velocities and accelerations
        if phase == 'ds':
            support_pose = np.hstack((
                self.plan[step_index]['ang'],
                self.plan[step_index]['pos']
            ))
            swing_pose = np.hstack((
                self.plan[step_index + 1]['ang'],
                self.plan[step_index + 1]['pos']
            ))
            zero_vel = np.zeros(6)
            zero_acc = np.zeros(6)
            
            support_foot= {
                    'pos': support_pose,
                    'vel': zero_vel,
                    'acc': zero_acc
                    },
            swing_foot= {
                    'pos': swing_pose,
                    'vel': zero_vel,
                    'acc': zero_acc
                    }
            if step_index%2==0:
                if self.first_swing=='lfoot':
                    l_foot_sum = np.vstack((l_foot_sum,support_foot))
                    r_foot_sum = np.vstack((r_foot_sum,swing_foot))
                if self.first_swing=='rfoot':
                    r_foot_sum = np.vstack((r_foot_sum,support_foot))
                    l_foot_sum = np.vstack((l_foot_sum,swing_foot))
            else:
                if self.first_swing=='lfoot':
                    l_foot_sum = np.vstack((l_foot_sum,swing_foot))
                    r_foot_sum = np.vstack((r_foot_sum,support_foot))
                if self.first_swing=='rfoot':
                    r_foot_sum = np.vstack((r_foot_sum,swing_foot))
                    l_foot_sum = np.vstack((l_foot_sum,support_foot))  
            
            continue #break the current execution, continute the next instance of "for" loop
            
    
        # get positions and angles for cubic interpolation
        start_pos  = self.plan[step_index - 1]['pos']
        target_pos = self.plan[step_index + 1]['pos']
        start_ang  = self.plan[step_index - 1]['ang']
        target_ang = self.plan[step_index + 1]['ang']

        # time variables
        t = time_in_step
        T = single_support_duration

        # cubic polynomial for position and angle
        A = - 2 / T**3
        B =   3 / T**2
        swing_pos     = start_pos + (target_pos - start_pos) * (    A * t**3 +     B * t**2)
        swing_vel     =             (target_pos - start_pos) * (3 * A * t**2 + 2 * B * t   ) / self.delta
        swing_acc     =             (target_pos - start_pos) * (6 * A * t    + 2 * B       ) / self.delta**2
        swing_ang_pos = start_ang + (target_ang - start_ang) * (    A * t**3 +     B * t**2)
        swing_ang_vel =             (target_ang - start_ang) * (3 * A * t**2 + 2 * B * t   ) / self.delta
        swing_ang_acc =             (target_ang - start_ang) * (6 * A * t    + 2 * B       ) / self.delta**2

        # quartic polynomial for vertical position
        A =   16 * self.step_height / T**4
        B = - 32 * self.step_height / T**3
        C =   16 * self.step_height / T**2
        swing_pos[2] =       A * t**4 +     B * t**3 +     C * t**2
        swing_vel[2] = ( 4 * A * t**3 + 3 * B * t**2 + 2 * C * t   ) / self.delta
        swing_acc[2] = (12 * A * t**2 + 6 * B * t    + 2 * C       ) / self.delta**2

        # support foot remains stationary
        support_pos = self.plan[step_index]['pos']
        support_ang = self.plan[step_index]['ang']
        zero_vel = np.zeros(3)
        zero_acc = np.zeros(3)

        # assemble pose, velocity, and acceleration for each foot
        support_data = {
            'pos': np.hstack((support_ang, support_pos)),
            'vel': np.hstack((np.zeros(3), zero_vel)),
            'acc': np.hstack((np.zeros(3), zero_acc))
        }

        swing_data = {
            'pos': np.hstack((swing_ang_pos, swing_pos)),
            'vel': np.hstack((swing_ang_vel, swing_vel)),
            'acc': np.hstack((swing_ang_acc, swing_acc))
        }
        #print("herre in pre trj")
        if support_foot=='lfoot':
               l_foot_sum = np.vstack((l_foot_sum,support_data))
        if swing_foot=='lfoot':
               l_foot_sum = np.vstack((l_foot_sum,swing_data))
        if support_foot=='rfoot':
               r_foot_sum = np.vstack((r_foot_sum,support_data))
        if swing_foot=='rfoot':
               r_foot_sum = np.vstack((r_foot_sum,swing_data))


    return {
        'lfoot': l_foot_sum,
        'rfoot': r_foot_sum
    }
  









  def generate_modified_swing_foot_trajectories_bounded(self, time, actual_pose, actual_vel, actual_acc,new_pose, planed_contact_pose):
        """
        Genera una traiettoria modificata per il piede in swing,
        imponendo un limite sullo scostamento tra new_pose e planed_contact_pose.
        """

        # ============================================================
        # 1) Definizione dei bound per scostamento MASSIMO consentito
        #    rispetto a planed_contact_pose
        #    - i primi 3 DOF si assumono essere l'orientamento (roll, pitch, yaw)
        #      e supponiamo di lavorare in radianti.
        #    - gli ultimi 3 DOF si assumono essere la posizione (x, y, z) in metri.
        # ============================================================
       
        #they should be updated automatically from centroidal MPC
        ORIENT_MAX_DIFF = 0.01
        ORIENT_MIN_DIFF = -ORIENT_MAX_DIFF

        POS_MAX_DIFF = 0.01
        POS_MIN_DIFF = -POS_MAX_DIFF

        # ============================================================
        # 2) Individuazione fase di ciclo e calcolo step_left
        #    (Stesso codice originale)
        # ============================================================
        phase_in_cycle = time % 100
        if phase_in_cycle < 70:
            current_phase = 'ss'
            steps_left = 70 - phase_in_cycle
        else:
            print("[generate_modified_feet_trajectories] ERRORE: siamo già in DS/fine ciclo.")
            self.modified_trajectory = None
            return

        start_time = time
        end_time = time + steps_left
        if steps_left <= 0:
            print(f"[generate_modified_feet_trajectories] steps_left={steps_left} => Non genero nulla.")
            self.modified_trajectory = None
            return

        # Time discretization
        time_steps = np.arange(start_time, end_time + 1)
        n_points = len(time_steps)
        duration = steps_left * self.delta  # self.delta è lo step di simulazione

        # Allochiamo array per la traiettoria
        pose_array = np.zeros((n_points, 6))
        vel_array  = np.zeros((n_points, 6))
        acc_array  = np.zeros((n_points, 6))

        # ============================================================
        # 3) Conversione in np.array
        # ============================================================
        actual_pose = np.asarray(actual_pose, dtype=float)
        actual_vel  = np.asarray(actual_vel,  dtype=float)
        actual_acc  = np.asarray(actual_acc,  dtype=float)
        new_pose    = np.asarray(new_pose,    dtype=float)
        planed_contact_pose = np.asarray(planed_contact_pose, dtype=float)

        # ============================================================
        # 4) "Clamping" di new_pose:
        #    Se la differenza rispetto a planed_contact_pose
        #    supera i limiti, la riportiamo al max/min ammissibile.
        # ============================================================
        for i in range(6):
            diff = new_pose[i] - planed_contact_pose[i]
            if i < 3:
                # Orientamento (roll, pitch, yaw)
                if diff > ORIENT_MAX_DIFF:
                    diff = ORIENT_MAX_DIFF
                elif diff < ORIENT_MIN_DIFF:
                    diff = ORIENT_MIN_DIFF
            else:
                # Posizione (x, y, z)
                if diff > POS_MAX_DIFF:
                    diff = POS_MAX_DIFF
                elif diff < POS_MIN_DIFF:
                    diff = POS_MIN_DIFF
            # Applico la differenza clampata
            new_pose[i] = planed_contact_pose[i] + diff

        # ============================================================
        # 5) Calcolo dei coefficienti quintici per le prime 5 componenti
        #    (stesso codice originale)
        # ============================================================
        quintic_coeffs = np.zeros((5, 6))  # c0..c5 per 5 DOF
        for i in range(5):
            x0 = actual_pose[i]
            v0 = actual_vel[i]
            a0 = actual_acc[i]
            xT = new_pose[i]   # posa finale (dopo clamping)
            vT = 0.0
            aT = 0.0

            c0 = x0
            c1 = v0
            c2 = a0/2.0

            R1 = xT - (c0 + c1*duration + c2*(duration**2))
            R2 = 0.0 - (c1 + 2*c2*duration)
            R3 = 0.0 - (2*c2)
            M = np.array([
                [duration**3,  duration**4,  duration**5],
                [3*duration**2, 4*duration**3, 5*duration**4],
                [6*duration,    12*duration**2, 20*duration**3]
            ])
            R = np.array([R1, R2, R3])
            c3, c4, c5 = np.linalg.solve(M, R)
            quintic_coeffs[i, :] = [c0, c1, c2, c3, c4, c5]

        # ============================================================
        # 6) Per la componente z usiamo una interpolazione lineare:
        #    - assumiamo new_pose[5] lo vogliamo a z = 0 a fine swing.
        #      In questo esempio però puoi anche usare new_pose[5] se preferisci.
        # ============================================================
        z0 = actual_pose[5]
        zT = 0.0  # Se vuoi forzare a 0, o potresti usare new_pose[5]

        # ============================================================
        # 7) Costruzione della traiettoria campione per campione
        # ============================================================
        for idx, t_step in enumerate(time_steps):
            t_rel = (t_step - start_time) * self.delta
            tau   = t_rel / duration if duration > 1e-9 else 1.0

            # prime 5 DOF: quintic
            xyzang     = np.zeros(5)
            xyzang_vel = np.zeros(5)
            xyzang_acc = np.zeros(5)

            for i in range(5):
                c0, c1, c2, c3, c4, c5 = quintic_coeffs[i]
                xyzang[i] = c0 + c1*t_rel + c2*(t_rel**2) \
                                + c3*(t_rel**3) + c4*(t_rel**4) + c5*(t_rel**5)

                xyzang_vel[i] = c1 + 2*c2*t_rel + 3*c3*(t_rel**2) \
                                + 4*c4*(t_rel**3) + 5*c5*(t_rel**4)

                xyzang_acc[i] = 2*c2 + 6*c3*t_rel + 12*c4*(t_rel**2) \
                                + 20*c5*(t_rel**3)

            # z lineare
            z_pos = z0 + (zT - z0) * tau
            z_vel = (zT - z0) / duration if duration > 1e-9 else 0.0
            z_acc = 0.0

            # salviamo nei vettori finali
            pose_array[idx, :5]  = xyzang
            pose_array[idx,  5]  = z_pos

            vel_array[idx, :5]   = xyzang_vel
            vel_array[idx,  5]   = z_vel

            acc_array[idx, :5]   = xyzang_acc
            acc_array[idx,  5]   = z_acc

        # ============================================================
        # 8) Forzo v=0 e a=0 all'ultimo campione per "fermare" la traiettoria
        # ============================================================
        vel_array[-1,:] = 0.0
        acc_array[-1,:] = 0.0

        # ============================================================
        # 9) Salvo la traiettoria generata in self.modified_trajectory
        # ============================================================
        self.modified_trajectory = {
            'active': True,
            'phase': current_phase,
            'times': time_steps,
            'poses': pose_array,
            'vels':  vel_array,
            'accs':  acc_array
        }

        # Debug info
        print(f"[generate_modified_feet_trajectories] Traiettoria da t={start_time} a t={end_time}, "
            f" con step_left={steps_left}. \n"
            f" - start_pose={actual_pose} \n"
            f" - end_pose={new_pose} (dopo clamping) \n"
            f" - z(0)={z0:.3f}, z(T)={zT:.3f} \n"
            f" - Forzo v=0 e a=0 all'ultimo campione.")





###This is the 'tested version'
  def generate_modified_swing_foot_trajectories(self, time, actual_pose, actual_vel, actual_acc, new_pose,planed_contact_pose):
    phase_in_cycle = time % 100
    if phase_in_cycle < 70:
        current_phase = 'ss'
        steps_left = 70 - phase_in_cycle
    else:
        print("[generate_modified_feet_trajectories] ERRORE: siamo già in DS/fine ciclo.")
        self.modified_trajectory = None
        return

    start_time = time
    end_time = time + steps_left
    if steps_left <= 0:
        print(f"[generate_modified_feet_trajectories] steps_left={steps_left} => Non genero nulla.")
        self.modified_trajectory = None
        return

    time_steps = np.arange(start_time, end_time + 1)
    n_points = len(time_steps)
    duration = steps_left * self.delta

    pose_array = np.zeros((n_points, 6))
    vel_array  = np.zeros((n_points, 6))
    acc_array  = np.zeros((n_points, 6))

    # Converto in np.array
    actual_pose = np.asarray(actual_pose)
    actual_vel  = np.asarray(actual_vel)
    actual_acc  = np.asarray(actual_acc)
    new_pose    = np.asarray(new_pose)

    # ----- Esempio: quintic per i primi 5 DOF, e lineare per z -----
    # (Codice simile a quanto mostrato prima, eventualmente ridotto.)

    # 1) Calcolo coefficienti quintici per i primi 5 DOF
    quintic_coeffs = np.zeros((5, 6))  # c0..c5
    for i in range(5):
        x0 = actual_pose[i]
        v0 = actual_vel[i]
        a0 = actual_acc[i]
        xT = new_pose[i]
        vT = 0.0
        aT = 0.0

        c0 = x0
        c1 = v0
        c2 = a0/2.0

        R1 = xT - (c0 + c1*duration + c2*(duration**2))
        R2 = 0.0 - (c1 + 2*c2*duration)
        R3 = 0.0 - (2*c2)
        M = np.array([
            [duration**3, duration**4, duration**5],
            [3*duration**2, 4*duration**3, 5*duration**4],
            [6*duration, 12*duration**2, 20*duration**3]
        ])
        R = np.array([R1, R2, R3])
        c3, c4, c5 = np.linalg.solve(M, R)
        quintic_coeffs[i,:] = [c0, c1, c2, c3, c4, c5]

    # 2) z lineare
    z0 = actual_pose[5]
    zT = 0

    # 3) Genero i campioni
    for idx, t_step in enumerate(time_steps):
        t_rel = (t_step - start_time)*self.delta
        tau = t_rel / duration if duration>1e-9 else 1.0

        # quintic su prime 5 componenti
        xyzang = np.zeros(5)
        xyzang_vel = np.zeros(5)
        xyzang_acc = np.zeros(5)

        for i in range(5):
            c0, c1, c2, c3, c4, c5 = quintic_coeffs[i]
            xyzang[i] = c0 + c1*t_rel + c2*(t_rel**2) + c3*(t_rel**3) \
                        + c4*(t_rel**4) + c5*(t_rel**5)

            xyzang_vel[i] = c1 + 2*c2*t_rel + 3*c3*(t_rel**2) \
                            + 4*c4*(t_rel**3) + 5*c5*(t_rel**4)

            xyzang_acc[i] = 2*c2 + 6*c3*t_rel + 12*c4*(t_rel**2) \
                            + 20*c5*(t_rel**3)

        # z lineare
        z_pos = z0 + (zT - z0)*tau
        z_vel = (zT - z0) / duration if duration>1e-9 else 0.0
        z_acc = 0.0

        # scrivo nei vettori finali
        pose_array[idx,:5] = xyzang
        pose_array[idx, 5] = z_pos

        vel_array[idx,:5] = xyzang_vel
        vel_array[idx, 5] = z_vel

        acc_array[idx,:5] = xyzang_acc
        acc_array[idx, 5] = z_acc

    # 4) Sovrascrivo velocità e accelerazioni finali con 0
    vel_array[-1,:] = 0.0
    acc_array[-1,:] = 0.0

    # Salvo la traiettoria
    self.modified_trajectory = {
        'active': True,
        'phase': current_phase,
        'times': time_steps,
        'poses': pose_array,
        'vels':  vel_array,
        'accs':  acc_array
    }

    print(f"[generate_modified_feet_trajectories] Traiettoria da t={start_time} a t={end_time}, "
          f" con step_left={steps_left}. \n"
          f" - start_pose={actual_pose} \n"
          f" - end_pose={new_pose} \n"
          f" - z(0)={z0:.3f}, z(T)={zT:.3f} \n"
          f" - Forzo v=0 e a=0 all'ultimo campione.")







  def generate_modified_feet_trajectories_at_time(self, time):
        """
        Ritorna (pos, vel, acc) dal buffer self.modified_trajectory
        corrispondente a 'time'. Se 'time' è fuori range, ritorna
        lo stato iniziale/finale o None, a tua scelta.
        """

        if (self.modified_trajectory is None) or (not self.modified_trajectory['active']):
            print("[generate_modified_feet_trajectories_at_time] ERRORE: Nessuna traiettoria modificata attiva.")
            return None

        times_arr = self.modified_trajectory['times']
        poses_arr = self.modified_trajectory['poses']
        vels_arr  = self.modified_trajectory['vels']
        accs_arr  = self.modified_trajectory['accs']

        if time < times_arr[0]:
            # Prima dell'inizio: ritorna i valori al primo indice
            return {
                'pos': poses_arr[0],
                'vel': vels_arr[0],
                'acc': accs_arr[0]
            }
        elif time > times_arr[-1]:
            # Oltre la fine: ritorna i valori all'ultimo indice
            return {
                'pos': poses_arr[-1],
                'vel': vels_arr[-1],
                'acc': accs_arr[-1]
            }
        else:
            # Nel range: cerchiamo l'indice esatto
            idxs = np.where(times_arr == time)[0]
            if len(idxs) == 0:
                # Se non lo trovi esattamente (p.es. times_arr salta certi step),
                # potresti fare un'interpolazione tra i due più vicini,
                # ma se times_arr è consecutivo, non succede mai.
                print(f"[generate_modified_feet_trajectories_at_time] ERRORE: time={time} non trovato.")
                return None
            idx = idxs[0]
            return {
                'pos': poses_arr[idx],
                'vel': vels_arr[idx],
                'acc': accs_arr[idx]
            }










