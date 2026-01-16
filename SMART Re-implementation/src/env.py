# src/env.py
import numpy as np

class CellularEnv:
    """
    Paper-accurate mmWave cellular environment
    Following SMART paper + Mismar et al. [15] channel model
    """
    def __init__(self, n_bs=2, n_ue=3, n_antennas=16, codebook_size=16,
                 noise_dbm=-110, gamma_min_db=-3, i_min_dbm=-110,
                 cell_radius=112, freq_ghz=28, max_power_dbm=32):
        
        self.n_bs = n_bs
        self.n_ue = n_ue
        self.M = n_antennas
        self.n_beams = codebook_size
        self.freq_ghz = freq_ghz
        self.wavelength = 3e8 / (freq_ghz * 1e9)
        self.cell_radius = cell_radius
        
        # Thresholds
        self.noise = 10 ** (noise_dbm / 10) / 1000
        self.gamma_min = 10 ** (gamma_min_db / 10)
        self.i_min = 10 ** (i_min_dbm / 10) / 1000
        self.penalty = -100
        
        # Power parameters
        self.power_step_db = 1.0
        self.max_power_dbm = max_power_dbm
        self.min_power_dbm = 0
        
        # Beamforming codebook
        self.codebook = self._generate_dft_codebook()
        
        # State variables
        self.bs_positions = None
        self.ue_positions = None
        self.channel_matrices = None
        self.prev_powers_dbm = None
        self.prev_beams = None
        
    def _generate_dft_codebook(self):
        """
        DFT codebook from El Ayach et al. [23]
        Eq. 2: a_t(x) = 1/√M [1, e^(jkd·x), ..., e^(j(M-1)kd·x)]
        where x = sin(φ) for steering angle φ
        """
        codebook = []
        
        # Steering angles covering [-π/2, π/2]
        angles = np.linspace(-np.pi/2, np.pi/2, self.n_beams)
        
        for phi in angles:  # phi is the steering angle
            d = self.wavelength / 2
            k = 2 * np.pi / self.wavelength
            
            # Array response: phases = kd·sin(φ) for each element
            # This is CORRECT per El Ayach Eq. 2 and 4
            phases = k * d * np.arange(self.M) * np.sin(phi)
            
            # Quantize phases (optional, for realistic phase shifters)
            r_bits = int(np.log2(self.n_beams))
            if r_bits > 0:
                num_levels = 2 ** r_bits
                phase_grid = np.linspace(0, 2*np.pi, num_levels, endpoint=False)
                
                quantized_phases = np.zeros(self.M)
                for m in range(self.M):
                    idx = np.argmin(np.abs(phase_grid - (phases[m] % (2*np.pi))))
                    quantized_phases[m] = phase_grid[idx]
                
                w = np.exp(1j * quantized_phases) / np.sqrt(self.M)
            else:
                w = np.exp(1j * phases) / np.sqrt(self.M)
            
            codebook.append(w)
        
        return np.array(codebook)
    
    def _generate_bs_positions(self):
        """Generate BS positions"""
        if self.n_bs == 2:
            isd = self.cell_radius * 2
            return [np.array([0.0, 0.0]), np.array([isd, 0.0])]
        else:
            positions = []
            isd = self.cell_radius * 2
            for i in range(self.n_bs):
                positions.append(np.array([i * isd, 0.0]))
            return positions
    
    
    def _generate_channel_saleh_valenzuela(self):
        """
        Saleh-Valenzuela channel model from Mismar et al. [15]
        Eq. 3: h = √(M/ρ) Σ_{i=1}^{N_cl} Σ_{ℓ=1}^{N_ray} α_iℓ a_t(sin(φ^t_iℓ))
        
        Per Mismar Table II: N_cl=8 (outdoor), N_ray=10
        """
        channels = np.zeros((self.n_bs, self.n_bs, self.n_ue, self.M), dtype=complex)
        
        # From Mismar et al. Table II
        N_cl = 8    # 8 clusters for outdoor mmWave
        N_ray = 10  # 10 rays per cluster
        
        for i in range(self.n_bs):
            for u in range(self.n_ue):
                ue_abs_pos = self.bs_positions[i] + self.ue_positions[i, u]
                
                for j in range(self.n_bs):
                    vec = ue_abs_pos - self.bs_positions[j]
                    dist = max(np.linalg.norm(vec), 1.0)
                    
                    # Path loss models from Mismar et al. [15]
                    if self.freq_ghz < 6:  # Sub-6GHz (voice)
                        # COST231 model
                        pl_db = 128.1 + 37.6 * np.log10(dist/1000)  # dist in meters
                    else:  # mmWave (28 GHz)
                        # Sulyman et al. model cited in Mismar
                        # Assume LOS-dominant for simplicity
                        pl_db = 72.0 + 29.2 * np.log10(dist)
                    
                    path_loss = 10 ** (-pl_db / 20)
                    
                    # Saleh-Valenzuela: Σ_{clusters} Σ_{rays} α_p a_t(sin(φ_p))
                    h = np.zeros(self.M, dtype=complex)
                    
                    for cluster in range(N_cl):
                        # Cluster AoD (azimuth)
                        phi_cluster = 2 * np.pi * np.random.rand()
                        
                        for ray in range(N_ray):
                            # Ray AoD within cluster (Laplacian angular spread)
                            # Standard deviation ~7.5° per Mismar simulations
                            phi_ray = phi_cluster + np.random.laplace(0, 0.13)
                            
                            # Complex path gain (Rayleigh distributed amplitude)
                            alpha = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
                            
                            # Normalize by number of paths
                            alpha *= np.sqrt(1 / (N_cl * N_ray))
                            
                            # Array response vector
                            a_t = self._array_response_ula(phi_ray)
                            
                            h += alpha * a_t
                    
                    # Apply path loss and normalization per Eq. 3
                    channels[j, i, u, :] = path_loss * np.sqrt(self.M) * h
        
        return channels

    def _array_response_ula(self, phi):
        """
        ULA array response from El Ayach et al. [23], Eq. 2
        a_t(x) = 1/√M [1, e^(jkd·x), ..., e^(j(M-1)kd·x)]
        where x = sin(φ)
        """
        d = self.wavelength / 2  # Half-wavelength spacing
        k = 2 * np.pi / self.wavelength
        
        # Phases: kd·sin(φ) per element
        phases = k * d * np.arange(self.M) * np.sin(phi)
        return np.exp(1j * phases) / np.sqrt(self.M)
    
    def reset(self):
        """Initialize episode"""
        self.bs_positions = self._generate_bs_positions()
        
        self.ue_positions = np.zeros((self.n_bs, self.n_ue, 2))
        for i in range(self.n_bs):
            for u in range(self.n_ue):
                r = self.cell_radius * 0.7 * np.sqrt(np.random.rand())
                theta = 2 * np.pi * np.random.rand()
                self.ue_positions[i, u] = [r * np.cos(theta), r * np.sin(theta)]
        
        self.channel_matrices = self._generate_channel_saleh_valenzuela()
        
        self.prev_powers_dbm = np.full((self.n_bs, self.n_ue), 20.0)
        self.prev_beams = np.random.randint(0, self.n_beams, (self.n_bs, self.n_ue))
        
        return self._get_states()
    
    def step(self, actions):
        """Execute actions"""
        # Update powers and beams
        for l in range(self.n_bs):
            for u in range(self.n_ue):
                self.prev_powers_dbm[l, u] += (2 * actions[l][u][0] - 1) * self.power_step_db
                self.prev_powers_dbm[l, u] = np.clip(
                    self.prev_powers_dbm[l, u],
                    self.min_power_dbm,
                    self.max_power_dbm
                )
                
                self.prev_beams[l, u] = (self.prev_beams[l, u] + 
                                         (2 * actions[l][u][1] - 1)) % self.n_beams
        
        # Power constraint
        for l in range(self.n_bs):
            total = np.sum(10 ** (self.prev_powers_dbm[l] / 10))
            max_total = (10 ** (self.max_power_dbm / 10)) * self.n_ue
            if total > max_total:
                scale = max_total / total
                for u in range(self.n_ue):
                    power_lin = 10 ** (self.prev_powers_dbm[l, u] / 10)
                    self.prev_powers_dbm[l, u] = 10 * np.log10(power_lin * scale)
        
        # Calculate rewards (per-UE constraint)
        # Calculate rewards (Paper Eq. 12: Product form)
        rewards = []
        inter_cell_ints = []

        for l in range(self.n_bs):
            product_term = 1.0
            all_conditions_met = True
            inter_ints_l = []
            
            for u in range(self.n_ue):
                sinr, inter_int = self._calculate_sinr(l, u)
                inter_ints_l.append(inter_int)
                
                # Paper Eq. 12: Check BOTH conditions
                if sinr < self.gamma_min or inter_int >= self.i_min:  
                    all_conditions_met = False
                
                product_term *= (1 + sinr)
            
            if all_conditions_met:
                rewards.append(product_term)
            else:
                rewards.append(self.penalty)
            
            inter_cell_ints.append(inter_ints_l)
        
        return self._get_states(), rewards, inter_cell_ints, False
    def get_interference_from_to(self, from_bs, to_bs, ue_idx):
        """
        Calculate interference FROM from_bs TO ue_idx in to_bs cell
        Used for experience sharing decision (Eq. 14)
        
        Returns:
            I^inter_from,to,ue in Watts
        """
        if from_bs == to_bs:
            return 0.0
        
        inter = 0.0
        for v in range(self.n_ue):
            power_v = 10 ** (self.prev_powers_dbm[from_bs, v] / 10) / 1000
            beam_v = self.codebook[self.prev_beams[from_bs, v]]
            h_cross = self.channel_matrices[from_bs, to_bs, ue_idx, :]
            inter += np.abs(np.vdot(beam_v, h_cross)) ** 2 * power_v
        
        return inter
    
    def _calculate_sinr(self, l, u):
        """Calculate SINR"""
        power = 10 ** (self.prev_powers_dbm[l, u] / 10) / 1000
        beam = self.codebook[self.prev_beams[l, u]]
        
        h = self.channel_matrices[l, l, u, :]
        signal = np.abs(np.vdot(beam, h)) ** 2 * power
        
        intra_int = 0
        for k in range(self.n_ue):
            if k != u:
                power_k = 10 ** (self.prev_powers_dbm[l, k] / 10) / 1000
                beam_k = self.codebook[self.prev_beams[l, k]]
                intra_int += np.abs(np.vdot(beam_k, h)) ** 2 * power_k
        
        inter_int = 0
        for j in range(self.n_bs):
            if j != l:
                for v in range(self.n_ue):
                    power_jv = 10 ** (self.prev_powers_dbm[j, v] / 10) / 1000
                    beam_jv = self.codebook[self.prev_beams[j, v]]
                    h_jl = self.channel_matrices[j, l, u, :]
                    inter_int += np.abs(np.vdot(beam_jv, h_jl)) ** 2 * power_jv
        
        sinr = signal / (intra_int + inter_int + self.noise + 1e-20)
        return sinr, inter_int
    
    def _get_states(self):
        """Get states for all agents"""
        states = []
        for l in range(self.n_bs):
            state_l = []
            for u in range(self.n_ue):
                state_u = np.array([
                    self.prev_powers_dbm[l, u] / self.max_power_dbm,
                    self.prev_beams[l, u] / self.n_beams,
                    self.ue_positions[l, u, 0] / self.cell_radius,
                    self.ue_positions[l, u, 1] / self.cell_radius
                ])
                state_l.append(state_u)
            states.append(np.array(state_l).flatten())
        return states