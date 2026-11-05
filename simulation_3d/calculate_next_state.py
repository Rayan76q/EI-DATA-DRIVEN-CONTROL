import numpy as np  
from simulation_3d.config import TE

def calculate_next_state(state_t, derivatives_t, type_of_bdd):#state_t = [x, y, z, v_x, v_y, v_z, phi, theta, psi, w_x, w_y, w_z] ; derivatives_t = [w_dot_x, w_dot_y, w_dot_z, a_x, a_y, a_z] if type_of_bdd ==2 else juste a_z
    x_t, y_t, z_t, v_x_t, v_y_t, v_z_t, phi_t, theta_t, psi_t, w_x_t, w_y_t, w_z_t = state_t
    
    if type_of_bdd == 2:
        w_dot_x_t, w_dot_y_t, w_dot_z_t, a_x_t, a_y_t, a_z_t = derivatives_t
        
        v_x_t2, v_y_t2, v_z_t2 = v_x_t + a_x_t * TE, v_y_t + a_y_t * TE, v_z_t + a_z_t * TE
        x_t2, y_t2, z_t2 = x_t + v_x_t * TE, y_t + v_y_t * TE, z_t + v_z_t * TE
        
        w_x_t2, w_y_t2, w_z_t2 = w_x_t + w_dot_x_t * TE, w_y_t + w_dot_y_t * TE, w_z_t + w_dot_z_t * TE
        
        phi_dot_t = w_x_t + np.sin(phi_t) * np.tan(theta_t) * w_y_t + np.cos(phi_t) * np.tan(theta_t) * w_z_t
        theta_dot_t = np.cos(phi_t) * w_y_t - np.sin(phi_t) * w_z_t
        psi_dot_t = (np.sin(phi_t) * w_y_t + np.cos(phi_t) * w_z_t)/np.cos(theta_t)
        
        phi_t2 = phi_t + phi_dot_t * TE
        theta_t2 = theta_t + theta_dot_t * TE
        psi_t2 = psi_t + psi_dot_t * TE        
        
        return (x_t2, y_t2, z_t2, v_x_t2, v_y_t2, v_z_t2, phi_t2, theta_t2, psi_t2, w_x_t2, w_y_t2, w_z_t2)
    
    elif type_of_bdd == 1:
        a_z_t = derivatives_t
        v_z_t2 = v_z_t + a_z_t * TE
        z_t2 = z_t + v_z_t * TE
        
        return (0, 0, z_t2, 0, 0, v_z_t2, 0, 0, 0, 0, 0, 0)
