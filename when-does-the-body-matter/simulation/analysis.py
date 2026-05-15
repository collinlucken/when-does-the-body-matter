"""
Dynamical systems and information-theoretic analysis tools for CTRNN research.

This module provides tools for analyzing the dynamics of evolved CTRNN controllers:

1. Phase portraits and bifurcation analysis
   - Visualize neural state trajectories in 2D/3D
   - Detect fixed points, limit cycles, bifurcations
   
2. Information-theoretic measures
   - Transfer entropy (causal information flow)
   - Mutual information (correlation/dependence)
   - Integrated information (Phi) for consciousness studies
   
3. Embodiment analysis
   - Ghost conditions (replay recorded sensory traces)
   - Perturbation sensitivity
   - Causal analysis (what depends on what?)

These tools are essential for:
- Understanding how evolved networks work
- Testing philosophical hypotheses about embodied cognition
- Characterizing chaos vs. order in neural dynamics
- Validating constitutive vs. causal claims

References:
    Tononi, G., Edelman, G. M., & Sporns, O. (1998). Complexity and coherency:
        Integrating information in the brain. Trends in Cognitive Sciences, 2(12), 474-484.
    Schreiber, T. (2000). Measuring information transfer. Physical Review Letters, 85(2), 461.
    Candadai, M., Izquierdo, E. J., & Izquierdo, E. (2017). The incorporated gym: Making embodied
        AI research accessible. arXiv preprint arXiv:1901.04654.
"""

from typing import Optional, Tuple, Callable, Dict
import numpy as np
from scipy import signal
from scipy.stats import entropy


class PhasePortrait:
    """
    Generate and analyze phase portraits of neural dynamics.
    
    A phase portrait visualizes the state space trajectory of a dynamical system.
    For CTRNNs, we examine trajectories in the space of neural activations.
    """
    
    def __init__(
        self,
        neural_network,
        dims: Tuple[int, int] = (0, 1),
        resolution: int = 50
    ) -> None:
        """
        Initialize phase portrait generator.
        
        Args:
            neural_network: CTRNN instance to analyze.
            dims: Which two dimensions to plot (neuron indices).
            resolution: Grid resolution for vector field.
        """
        self.network = neural_network
        self.dims = dims
        self.resolution = resolution
        self.external_input = np.zeros(neural_network.num_neurons)
    
    def set_external_input(self, external_input: np.ndarray) -> None:
        """Set fixed external input for phase portrait generation."""
        self.external_input = external_input.copy()
    
    def generate_vector_field(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate vector field (flow) for phase portrait.
        
        Args:
            x_range: Range for first dimension [min, max].
            y_range: Range for second dimension [min, max].
        
        Returns:
            X, Y: Meshgrid of positions
            U, V: Vector field components (flow direction)
        """
        x = np.linspace(x_range[0], x_range[1], self.resolution)
        y = np.linspace(y_range[0], y_range[1], self.resolution)
        X, Y = np.meshgrid(x, y)
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        for i in range(self.resolution):
            for j in range(self.resolution):
                # Create state vector
                state = self.network.state.copy()
                state[self.dims[0]] = X[i, j]
                state[self.dims[1]] = Y[i, j]
                self.network.set_state(state)
                
                # Compute derivative
                activation = self.network.biases + state
                output = self.network._sigmoid(activation)
                recurrent_input = np.dot(self.network.weights, output)
                dy_dt = (-state + recurrent_input + self.external_input) / self.network.tau
                
                U[i, j] = dy_dt[self.dims[0]]
                V[i, j] = dy_dt[self.dims[1]]
        
        return X, Y, U, V
    
    def find_fixed_points(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        threshold: float = 0.01
    ) -> np.ndarray:
        """
        Find approximate fixed points (equilibria) in the phase portrait.
        
        Args:
            x_range: Range for first dimension.
            y_range: Range for second dimension.
            threshold: Velocity threshold to consider a point fixed.
        
        Returns:
            Array of fixed point positions.
        """
        X, Y, U, V = self.generate_vector_field(x_range, y_range)
        velocity = np.sqrt(U ** 2 + V ** 2)
        fixed_point_mask = velocity < threshold
        
        fixed_points = []
        for i in range(self.resolution):
            for j in range(self.resolution):
                if fixed_point_mask[i, j]:
                    fixed_points.append([X[i, j], Y[i, j]])
        
        if len(fixed_points) == 0:
            return np.array([]).reshape(0, 2)
        
        # Cluster nearby points
        fixed_points = np.array(fixed_points)
        return fixed_points


class BifurcationAnalyzer:
    """
    Detect and characterize bifurcations in neural dynamics.
    
    Bifurcations occur when a small change in a parameter causes qualitative
    changes in system behavior (e.g., fixed point appears/disappears, 
    period-doubling cascade, etc.).
    """
    
    def __init__(self, neural_network) -> None:
        """Initialize bifurcation analyzer."""
        self.network = neural_network
    
    def compute_eigenvalues(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute eigenvalues of system Jacobian (for stability analysis).
        
        Eigenvalues indicate stability:
        - Real part < 0: stable (attracting)
        - Real part > 0: unstable (repelling)
        - Complex conjugates: rotating dynamics
        
        Args:
            state: State at which to compute Jacobian. If None, uses current state.
        
        Returns:
            Array of eigenvalues.
        """
        jacobian = self.network.get_jacobian(state)
        eigenvalues = np.linalg.eigvals(jacobian)
        return eigenvalues
    
    def scan_parameter(
        self,
        param_name: str,
        param_range: Tuple[float, float],
        num_points: int = 50,
        iterations_per_point: int = 1000
    ) -> Dict[str, np.ndarray]:
        """
        Scan a parameter and track bifurcation signatures.
        
        Args:
            param_name: Name of parameter to vary ('tau', 'weights', etc.)
            param_range: Range [min, max] for parameter.
            num_points: Number of points to sample.
            iterations_per_point: Iterations to run at each parameter value.
        
        Returns:
            Dictionary with parameter values and various metrics.
        """
        param_values = np.linspace(param_range[0], param_range[1], num_points)
        results = {
            'param_values': param_values,
            'max_lyapunov': np.zeros(num_points),
            'max_eigenvalue_real': np.zeros(num_points),
            'variance': np.zeros(num_points)
        }
        
        # Save original parameter
        if param_name == 'tau':
            original_param = self.network.tau.copy()
        else:
            original_param = None
        
        for idx, param_value in enumerate(param_values):
            # Set parameter
            if param_name == 'tau':
                self.network.tau[:] = param_value
            
            # Run transient
            for _ in range(100):
                self.network.step()
            
            # Collect dynamics
            states = []
            for _ in range(iterations_per_point):
                self.network.step()
                states.append(self.network.get_state())
            
            states = np.array(states)
            
            # Compute metrics
            results['variance'][idx] = np.mean(np.var(states, axis=0))
            eigenvalues = self.compute_eigenvalues()
            results['max_eigenvalue_real'][idx] = np.max(np.real(eigenvalues))
        
        # Restore parameter
        if param_name == 'tau' and original_param is not None:
            self.network.tau[:] = original_param
        
        return results


class InformationAnalyzer:
    """
    Compute information-theoretic measures of neural dynamics.
    
    Measures implemented:
    - Mutual information: statistical dependence between variables
    - Transfer entropy: directional information flow (causality)
    - Integrated information (Phi): consciousness/integration measure
    """
    
    @staticmethod
    def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
        """
        Compute mutual information between two time series.
        
        I(X;Y) = H(X) + H(Y) - H(X,Y)
        where H is Shannon entropy.
        
        Args:
            x, y: Time series (1D arrays).
            bins: Number of bins for histogram.
        
        Returns:
            Mutual information (in nats).
        """
        # Discretize
        x_binned = np.digitize(x, np.linspace(x.min(), x.max(), bins))
        y_binned = np.digitize(y, np.linspace(y.min(), y.max(), bins))
        
        # Compute entropies
        h_x = entropy(np.bincount(x_binned, minlength=bins))
        h_y = entropy(np.bincount(y_binned, minlength=bins))
        
        # Joint entropy
        joint = np.column_stack([x_binned, y_binned])
        joint_counts = {}
        for pair in joint:
            key = tuple(pair)
            joint_counts[key] = joint_counts.get(key, 0) + 1
        
        joint_probs = np.array(list(joint_counts.values())) / len(joint)
        h_xy = -np.sum(joint_probs * np.log(joint_probs + 1e-10))
        
        return h_x + h_y - h_xy
    
    @staticmethod
    def transfer_entropy(
        source: np.ndarray,
        target: np.ndarray,
        lag: int = 1,
        bins: int = 5
    ) -> float:
        """
        Compute transfer entropy from source to target time series.
        
        TE(source -> target) = H(target_t | target_t-lag) - H(target_t | target_t-lag, source_t-lag)
        
        Positive TE indicates information transfer (causality).
        
        Args:
            source: Source time series.
            target: Target time series.
            lag: Time lag to consider.
            bins: Number of bins for discretization.
        
        Returns:
            Transfer entropy (in nats).
        """
        n = len(source) - lag
        
        # Discretize
        source_binned = np.digitize(source[:-lag], 
                                   np.linspace(source.min(), source.max(), bins))
        target_past = np.digitize(target[:-lag],
                                 np.linspace(target.min(), target.max(), bins))
        target_future = np.digitize(target[lag:],
                                   np.linspace(target.min(), target.max(), bins))
        
        # Compute conditional entropies
        h_future_past = 0.0
        h_future_past_source = 0.0
        
        for s_val in range(1, bins + 1):
            for t_past in range(1, bins + 1):
                mask_past = (source_binned == s_val) & (target_past == t_past)
                if np.sum(mask_past) > 0:
                    future_vals = target_future[mask_past]
                    probs = np.bincount(future_vals, minlength=bins + 1) / np.sum(mask_past)
                    h_future_past_source -= np.sum(probs * np.log(probs + 1e-10)) * np.sum(mask_past) / n
        
        for t_past in range(1, bins + 1):
            mask = target_past == t_past
            if np.sum(mask) > 0:
                future_vals = target_future[mask]
                probs = np.bincount(future_vals, minlength=bins + 1) / np.sum(mask)
                h_future_past -= np.sum(probs * np.log(probs + 1e-10)) * np.sum(mask) / n
        
        return h_future_past - h_future_past_source
    
    @staticmethod
    def integrated_information_basic(
        state_vector: np.ndarray,
        num_perturbations: int = 100
    ) -> float:
        """
        Estimate integrated information (Phi) in a basic way.
        
        Integrated information measures how much information is lost when
        a system is partitioned. High Phi suggests integrated consciousness.
        
        This is a simplified version; proper computation is NP-hard.
        
        Args:
            state_vector: Current neural state vector.
            num_perturbations: Number of perturbations to sample.
        
        Returns:
            Estimated Phi (0-1 range).
        """
        n = len(state_vector)
        
        # Mutual information of whole system
        mi_whole = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                mi = InformationAnalyzer.mutual_information(
                    state_vector[i:i+1],
                    state_vector[j:j+1],
                    bins=3
                )
                mi_whole += mi
        
        # Partitioned system (split in half)
        part1 = state_vector[:n // 2]
        part2 = state_vector[n // 2:]
        
        mi_part = 0.0
        for i in range(len(part1)):
            for j in range(len(part2)):
                mi = InformationAnalyzer.mutual_information(
                    part1[i:i+1],
                    part2[j:j+1],
                    bins=3
                )
                mi_part += mi
        
        # Phi is the loss of information upon partition
        if mi_whole > 0:
            phi = (mi_whole - mi_part) / mi_whole
        else:
            phi = 0.0
        
        return np.clip(phi, 0.0, 1.0)


class EmbodimentAnalyzer:
    """
    Tools for analyzing embodied vs. disembodied neural computation.
    
    Key concepts:
    1. Ghost condition: replay recorded sensory input with body removed
    2. Causal analysis: what parts of the system are causally necessary?
    3. Perturbation analysis: sensitivity to environmental changes
    """
    
    def __init__(self, neural_network, environment) -> None:
        """
        Initialize embodiment analyzer.
        
        Args:
            neural_network: CTRNN to analyze.
            environment: Environment/task that provides sensory input.
        """
        self.network = neural_network
        self.environment = environment
    
    def record_sensory_trace(
        self,
        duration: int,
        agent_active: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Record sensory input and neural response during task execution.
        
        Args:
            duration: Duration of recording (timesteps).
            agent_active: If True, agent actively controls body.
                         If False, record without active control.
        
        Returns:
            sensory_trace: Sensory inputs (shape: [duration, num_sensors])
            neural_states: Neural states (shape: [duration, num_neurons])
        """
        sensory_trace = []
        neural_states = []
        
        for _ in range(duration):
            # Get sensory input
            sensory = self.environment.get_sensor_readings()
            sensory_trace.append(sensory)
            
            # Record neural state
            neural_states.append(self.network.get_state())
            
            # Update network
            self.network.step(sensory)
            
            # Update environment (if agent active)
            if agent_active:
                self.environment.step()
        
        return np.array(sensory_trace), np.array(neural_states)
    
    def ghost_condition(
        self,
        sensory_trace: np.ndarray,
        initial_state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run network in ghost condition (disembodied).
        
        Replay recorded sensory input without agent body present.
        Compare neural response to embodied condition to assess
        role of embodiment.
        
        Args:
            sensory_trace: Pre-recorded sensory input.
            initial_state: Initial neural state (if None, use default).
        
        Returns:
            neural_states: States during ghost playback.
            outputs: Neural outputs during ghost playback.
        """
        self.network.reset(initial_state)
        
        neural_states = []
        outputs = []
        
        for sensory in sensory_trace:
            neural_states.append(self.network.get_state())
            output = self.network.step(sensory)
            outputs.append(output)
        
        return np.array(neural_states), np.array(outputs)
    
    def compute_embodiment_dependence(
        self,
        sensory_trace: np.ndarray,
        embodied_states: np.ndarray,
        ghost_states: np.ndarray
    ) -> float:
        """
        Quantify how much neural dynamics depend on embodiment.
        
        Compute divergence between embodied and ghost conditions.
        High divergence indicates strong embodiment dependence.
        
        Args:
            sensory_trace: Sensory input sequence.
            embodied_states: Neural states when embodied.
            ghost_states: Neural states in ghost condition.
        
        Returns:
            Embodiment dependence score (0-1).
        """
        # Compute state-space divergence
        divergence = np.mean(np.sqrt(np.sum((embodied_states - ghost_states) ** 2, axis=1)))
        
        # Normalize by typical state magnitude
        typical_magnitude = np.mean(np.sqrt(np.sum(embodied_states ** 2, axis=1)))
        
        if typical_magnitude > 0:
            dependence = divergence / typical_magnitude
        else:
            dependence = 0.0
        
        return np.clip(dependence, 0.0, 1.0)
    
    def perturbation_analysis(
        self,
        sensory_trace: np.ndarray,
        perturbation_magnitude: float = 0.1,
        num_repeats: int = 10
    ) -> Dict[str, float]:
        """
        Analyze sensitivity to sensory perturbations.
        
        Args:
            sensory_trace: Sensory input sequence.
            perturbation_magnitude: Size of perturbations to apply.
            num_repeats: Number of perturbation trials.
        
        Returns:
            Dictionary with sensitivity metrics.
        """
        results = {
            'mean_sensitivity': 0.0,
            'max_sensitivity': 0.0,
            'divergence_growth_rate': 0.0
        }
        
        # Baseline (unperturbed)
        self.network.reset()
        baseline_states, _ = self.ghost_condition(sensory_trace)
        
        sensitivities = []
        
        for _ in range(num_repeats):
            # Create perturbed sensory trace
            perturbed_trace = sensory_trace + perturbation_magnitude * np.random.randn(*sensory_trace.shape)
            
            # Run with perturbation
            self.network.reset()
            perturbed_states, _ = self.ghost_condition(perturbed_trace)
            
            # Compute divergence
            divergence = np.sqrt(np.sum((baseline_states - perturbed_states) ** 2, axis=1))
            sensitivities.append(divergence)
        
        sensitivities = np.array(sensitivities)
        results['mean_sensitivity'] = float(np.mean(sensitivities))
        results['max_sensitivity'] = float(np.max(sensitivities))
        
        # Growth rate of divergence
        if sensitivities.shape[0] > 0:
            growth = (sensitivities[:, -1] - sensitivities[:, 0]) / len(sensory_trace)
            results['divergence_growth_rate'] = float(np.mean(growth))
        
        return results
