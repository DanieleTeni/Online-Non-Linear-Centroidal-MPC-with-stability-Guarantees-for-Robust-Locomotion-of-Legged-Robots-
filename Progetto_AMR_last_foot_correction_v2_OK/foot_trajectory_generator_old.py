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
  














  def generate_modified_swing_foot_trajectories(self,time,actual_pose,actual_vel,actual_acc,new_pose):
        """
        Genera un'intera traiettoria discreta (pos, vel, acc) dal tempo (time+1)
        fino alla fine della single support phase. 
        - Parte da (actual_pose, actual_vel, actual_acc)
        - Arriva in (new_pose, 0, 0)
        Assicurandosi che la z del piede sia sempre > 0 fino allo step finale.
        """

        phase_in_cycle = time % 100
        # Controlliamo se siamo in single support
        if phase_in_cycle < 70:
            current_phase = 'ss'
            steps_left = 70 - phase_in_cycle
        else:
            print("[generate_modified_feet_trajectories] ERRORE: siamo in DS o fine ciclo.")
            current_phase = 'ds'
            steps_left = 0

        start_time = time 
        end_time = time + steps_left

        if steps_left <= 0:
            print(f"[generate_modified_feet_trajectories] steps_left={steps_left} => Non genero nulla.")
            self.modified_trajectory = None
            return

        # Prepariamo un array di time-step (interi)
        time_steps = np.arange(start_time, end_time + 1)# end-time included
        n_points = len(time_steps)
        duration = (steps_left) * self.delta  # tempo in secondi di SS rimanente

        # Creiamo array vuoti per pos,vel,acc
        pose_array = np.zeros((n_points, 6))
        vel_array  = np.zeros((n_points, 6))
        acc_array  = np.zeros((n_points, 6))

        actual_pose = np.asarray(actual_pose)
        actual_vel  = np.asarray(actual_vel)
        actual_acc  = np.asarray(actual_acc)
        new_pose    = np.asarray(new_pose)

        # -- 1) Prepariamo i polinomi per x,y,angoli (5 dof). --
        # Li facciamo con un quintic, o un cubic "4 boundary conditions".
        # Per semplicità, userò un quintic con:
        #    x(0)=x0, x'(0)=v0, x''(0)=a0
        #    x(T)=xT, x'(T)=0, x''(T)=0
        # Ma SOLO per i dof [0..4], *escludendo la z (indice 5)*,
        # perché la z la trattiamo con uno schema custom per evitare z<=0.

        # Calcoliamo i coefficienti quintici dof per dof
        quintic_coeffs = np.zeros((5,6))  # (dof=5, c0..c5)

        for i in range(5):  # [angX,angY,angZ,x,y], la z è i=5
            x0 = actual_pose[i]
            v0 = actual_vel[i]
            a0 = actual_acc[i]
            xT = new_pose[i]
            vT = 0.0
            aT = 0.0

            c0 = x0
            c1 = v0
            c2 = a0/2.0

            # Sistemi:
            # x(T)=c0+c1T+c2T^2+c3T^3+c4T^4+c5T^5 => xT
            # x'(T)=c1+2c2T+3c3T^2+4c4T^3+5c5T^4 => 0
            # x''(T)=2c2+6c3T+12c4T^2+20c5T^3 => 0
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

        # -- 2) Gestiamo la Z con un polinomio quartic "lift" per evitare z<=0 --
        #   Vogliamo che:
        #    - z(0) = actual_pose[5]
        #    - z(T) = new_pose[5]
        #    - la forma in mezzo sia "a campana", cioè si sollevi e non scenda sotto 0.
        #   Se startZ o endZ sono <= 0, imponiamo un offset per tenerlo > 0.
        startZ = actual_pose[5]
        if startZ <= 0:
            startZ = 0.01  # piccolo offset
        endZ = new_pose[5]
        if endZ <= 0:
            endZ = 0.01

        # Coeff. quartic che sia: 
        # z(t) = poly(t) con z(0)=startZ, z(T)=endZ,
        # e "booster" step_height nel mezzo in modo che non scenda sotto startZ.
        # Ad esempio, come nel tuo foot code (16, -32, 16) per l' "arco" in [0,T].
        #
        # Facciamo un'alpha(t) = (t/T)^2 * (3 - 2(t/T)) -> classica cubic 0->1
        # E in parallelo aggiungiamo un "lift" quartic:
        # lift(tau)= step_height*(16*tau^4 - 32*tau^3 + 16*tau^2),
        # con tau=t/T in [0..1].
        # Che è 0 a tau=0 e tau=1, e positiva in mezzo.
        # -> z(t) =  z_start + (z_end - z_start)*alpha(tau) + lift(tau).
        # 
        # Se vuoi allineare anche vel,acc iniziali/finali, le cose si complicano:
        # potresti dover fare un polinomio quintico pure su z. 
        # Ma l'utente dice: "non usare vincoli aggiuntivi, voglio un interpolation
        # che non vada mai sotto 0". Allora questa forma 'lift' è un trucco comune.

        def alpha_cubic(tau):
            # 3tau^2 - 2tau^3
            return 3*(tau**2) - 2*(tau**3)

        def alpha_cubic_dot(tau):
            # d/dtau di [3tau^2 - 2tau^3] = 6tau - 6tau^2
            return 6*tau - 6*(tau**2)

        def alpha_cubic_ddot(tau):
            # d/dtau(6tau - 6tau^2) = 6 - 12tau
            return 6 - 12*tau

        def lift_quartic(tau):
            # step_height*(16tau^4 - 32tau^3 +16tau^2)
            return self.step_height*(16*(tau**4) - 32*(tau**3) + 16*(tau**2))

        def lift_quartic_dot(tau):
            # derivative wrt tau => step_height*(64tau^3 - 96tau^2 + 32tau)
            return self.step_height*(64*(tau**3) - 96*(tau**2) + 32*tau)

        def lift_quartic_ddot(tau):
            # step_height*(192tau^2 -192tau +32)
            return self.step_height*(192*(tau**2) - 192*tau + 32)

        # 3) Calcoliamo i sample su [start_time..end_time]
        for idx, t_step in enumerate(time_steps):
            # t_rel in secondi
            t_rel = (t_step - start_time)*self.delta
            tau = t_rel / duration if duration>1e-9 else 1.0  # normalizzato [0..1]

            # (a) x,y,ang = quintic
            xyzang = np.zeros(5)
            xyzang_vel = np.zeros(5)
            xyzang_acc = np.zeros(5)
            for i in range(5):
                c0, c1, c2, c3, c4, c5 = quintic_coeffs[i,:]
                # pos
                xyzang[i] = (c0
                             + c1*t_rel
                             + c2*(t_rel**2)
                             + c3*(t_rel**3)
                             + c4*(t_rel**4)
                             + c5*(t_rel**5))
                # vel
                xyzang_vel[i] = (c1
                                 + 2*c2*t_rel
                                 + 3*c3*(t_rel**2)
                                 + 4*c4*(t_rel**3)
                                 + 5*c5*(t_rel**4))
                # acc
                xyzang_acc[i] = (2*c2
                                 + 6*c3*t_rel
                                 + 12*c4*(t_rel**2)
                                 + 20*c5*(t_rel**3))

            # (b) z = "lift quartic"
            # z(t)= startZ + (endZ - startZ)*alpha_cubic(tau) + lift_quartic(tau)
            a_val   = alpha_cubic(tau)
            a_dot   = alpha_cubic_dot(tau)
            a_ddot  = alpha_cubic_ddot(tau)
            l_val   = lift_quartic(tau)
            l_dot   = lift_quartic_dot(tau)
            l_ddot  = lift_quartic_ddot(tau)

            z_pos = startZ + (endZ - startZ)*a_val + l_val
            z_vel = ((endZ - startZ)*a_dot + l_dot) / duration  # chain rule per dtau/dt = 1/duration
            z_acc = ((endZ - startZ)*a_ddot + l_ddot) / (duration**2)

            # Salviamo
            # Indici [0..2]=angX,angY,angZ, [3..4]=x,y, [5]=z
            pose_array[idx, :5] = xyzang
            vel_array[idx,  :5] = xyzang_vel
            acc_array[idx,  :5] = xyzang_acc

            pose_array[idx, 5] = z_pos
            vel_array[idx,   5] = z_vel
            acc_array[idx,   5] = z_acc

            # In questo modo la z risulterà sempre > 0 se step_height>0,
            # salvo casi estremi in cui (startZ + lift) < 0.
            # Ma abbiamo forzato startZ>=0.01, endZ>=0.01 => in mezzo è sempre > 0.

        # Infine, salviamo tutto in self.modified_trajectory
        self.modified_trajectory = {
            'active': True,
            'phase': current_phase,
            'times': time_steps,
            'poses': pose_array,
            'vels':  vel_array,
            'accs':  acc_array
        }

        print(f"[generate_modified_feet_trajectories] Generata traiettoria da t={start_time} a t={end_time} "
              f"(fase={current_phase})."
              f"\n - start pose={actual_pose}, vel={actual_vel}, acc={actual_acc}"
              f"\n - end   pose={new_pose} (vel=0, acc=0)"
              f"\n - z start={pose_array[0,5]:.3f}, z end={pose_array[-1,5]:.3f}"
              f"\n len_trajcetory:{len(self.modified_trajectory)}")




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
