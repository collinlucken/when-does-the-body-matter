"""
Continuous-Time Recurrent Neural Network (CTRNN) implementation for embodied cognition research.

This module implements CTRNNs as described in:
  Beer, R. D. (1995). On the dynamics of small continuous-time recurrent neural networks.
  Adaptive Behavior, 3(4), 469-509.

CTRNNs are used extensively in evolutionary robotics to model neural control systems for
autonomous agents. The key advantages are:
1. Continuous dynamics without discrete time steps
2. Interpretable as a nonlinear dynamical system
3. Amenable to bifurcation analysis and other dynamical tools
4. Biologically plausible integration (Euler method approximates dendritic time constants)
"""

from typing import Optional, Tuple
import numpy as np


class CTRNN:
    """
    Continuous-Time Recurrent Neural Network.
    
    The dynamics are governed by:
        tau_i * dy_i/dt = -y_i + sum_j(w_ij * sigma(y_j + theta_j)) + I_i
    
    where:
        y_i: output of neuron i (in [0, 1] after sigmoid activation)
        tau_i: time constant of neuron i (controls integration rate)
        w_ij: connection weight from neuron j to neuron i
        theta_j: bias (threshold/DC offset) of neuron j
        sigma: sigmoid activation function
        I_i: external input to neuron i
    
    The state y_i represents the membrane potential, and the output is obtained by
    passing it through a sigmoid: output_i = 1 / (1 + exp(-(y_i + bias_i)))
    
    References:
        Beer (1995) provides the mathematical foundations and demonstrates how CTRNNs
        can exhibit complex dynamical behaviors including chaos, limit cycles, and
        bifurcations relevant to embodied cognition.
    """
    
    def __init__(
        self,
        num_neurons: int,
        time_constants: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        biases: Optional[np.ndarray] = None,
        gains: Optional[np.ndarray] = None,
        step_size: float = 0.01,
        center_crossing: bool = True
    ) -> None:
        """
        Initialize a CTRNN.
        
        Args:
            num_neurons: Number of neurons in the network.
            time_constants: Neural time constants tau (shape: [num_neurons]).
                           If None, defaults to ones (tau=1.0).
            weights: Recurrent weight matrix w_ij (shape: [num_neurons, num_neurons]).
                    If None, defaults to zeros.
            biases: Neural bias/threshold values theta (shape: [num_neurons]).
                   If None, defaults to zeros.
            gains: Output gains (multiplicative factors before sigmoid).
                  Shape: [num_neurons]. If None, defaults to ones.
            step_size: Integration step size for Euler method (dt).
            center_crossing: If True, use centered sigmoid with output in [-1, 1].
                            If False, use standard sigmoid with output in [0, 1].
                            Default True follows Beer (1995) convention.
        """
        self.num_neurons = num_neurons
        self.step_size = step_size
        self.center_crossing = center_crossing
        
        # Initialize parameters with sensible defaults
        self.tau = time_constants if time_constants is not None else np.ones(num_neurons)
        self.weights = weights if weights is not None else np.zeros((num_neurons, num_neurons))
        self.biases = biases if biases is not None else np.zeros(num_neurons)
        self.gains = gains if gains is not None else np.ones(num_neurons)
        
        # Validate shapes
        assert self.tau.shape == (num_neurons,), f"tau shape mismatch: {self.tau.shape}"
        assert self.weights.shape == (num_neurons, num_neurons), f"weights shape mismatch: {self.weights.shape}"
        assert self.biases.shape == (num_neurons,), f"biases shape mismatch: {self.biases.shape}"
        assert self.gains.shape == (num_neurons,), f"gains shape mismatch: {self.gains.shape}"
        
        # Neural state (y_i in Beer 1995 notation)
        self.state = np.zeros(num_neurons)
        
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.
        
        If center_crossing=True: sigma(x) = 2/(1+exp(-x)) - 1  (output in [-1, 1])
        If center_crossing=False: sigma(x) = 1/(1+exp(-x))     (output in [0, 1])
        
        Args:
            x: Input array.
            
        Returns:
            Sigmoid activation of x.
        """
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        if self.center_crossing:
            return 2.0 * sigmoid - 1.0
        return sigmoid
    
    def step(self, external_inputs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform one integration step using Euler method.
        
        Implements: y_{n+1} = y_n + (dt/tau_i) * (-y_n + f(y_n) + I_n)
        where f(y_n) represents the nonlinear recurrent interactions.
        
        Args:
            external_inputs: External input vector I (shape: [num_neurons]).
                            If None, defaults to zeros.
        
        Returns:
            Output vector (sigmoid-activated state).
        """
        if external_inputs is None:
            external_inputs = np.zeros(self.num_neurons)
        
        # Compute nonlinear activation of current state with biases
        activation = self.biases + self.state  # y_i + theta_i
        output = self._sigmoid(activation)  # sigma(y_i + theta_i)
        
        # Compute recurrent input: sum_j(w_ij * sigma(y_j + theta_j))
        recurrent_input = np.dot(self.weights, output)
        
        # Euler integration step for each neuron
        # dy_i/dt = (-y_i + recurrent_input_i + external_input_i) / tau_i
        dy_dt = (-self.state + recurrent_input + external_inputs) / self.tau
        self.state += self.step_size * dy_dt
        
        return output
    
    def reset(self, initial_state: Optional[np.ndarray] = None) -> None:
        """
        Reset neural state to initial condition.
        
        Args:
            initial_state: Initial state vector (shape: [num_neurons]).
                          If None, resets to zeros.
        """
        if initial_state is None:
            self.state = np.zeros(self.num_neurons)
        else:
            assert initial_state.shape == (self.num_neurons,)
            self.state = initial_state.copy()
    
    def get_output(self) -> np.ndarray:
        """
        Get current output without advancing time.
        
        Returns:
            Sigmoid-activated current state.
        """
        activation = self.biases + self.state
        return self._sigmoid(activation)
    
    def set_state(self, state: np.ndarray) -> None:
        """
        Directly set internal state (for analysis/perturbation studies).
        
        Args:
            state: State vector to set.
        """
        assert state.shape == (self.num_neurons,)
        self.state = state.copy()
    
    def get_state(self) -> np.ndarray:
        """Get current internal state."""
        return self.state.copy()
    
    def run(
        self,
        inputs: np.ndarray,
        reset_state: bool = True,
        initial_state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the network for a sequence of inputs.
        
        Args:
            inputs: Input sequence (shape: [timesteps, num_neurons]).
            reset_state: If True, reset state before running.
            initial_state: Initial state (only used if reset_state=True).
        
        Returns:
            outputs: Output sequence (shape: [timesteps, num_neurons])
            states: State sequence (shape: [timesteps, num_neurons])
        """
        if reset_state:
            self.reset(initial_state)
        
        timesteps = inputs.shape[0]
        outputs = np.zeros((timesteps, self.num_neurons))
        states = np.zeros((timesteps, self.num_neurons))
        
        for t in range(timesteps):
            outputs[t] = self.step(inputs[t])
            states[t] = self.state.copy()
        
        return outputs, states
    
    def get_jacobian(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute Jacobian of the system at a given state point.
        
        Useful for stability analysis and bifurcation detection.
        The Jacobian describes how perturbations propagate locally.
        
        Args:
            state: State at which to compute Jacobian. If None, uses current state.
        
        Returns:
            Jacobian matrix (shape: [num_neurons, num_neurons]).
        """
        if state is None:
            state = self.state
        
        # Compute sigma(y + theta) and its derivative at current point
        activation = self.biases + state
        sigmoid_output = self._sigmoid(activation)
        
        # Derivative of sigmoid: sigma'(x) = sigma(x) * (1 - sigma(x)) for standard
        # For center-crossing: sigma'(x) = 2 * sigma(x) * (1 - sigma(x))
        if self.center_crossing:
            sigmoid_deriv = 2.0 * sigmoid_output * (1.0 - sigmoid_output)
        else:
            sigmoid_deriv = sigmoid_output * (1.0 - sigmoid_output)
        
        # Jacobian: J_ij = (dt/tau_i) * (delta_ij * (-1) + w_ij * sigma'_j)
        jacobian = np.zeros((self.num_neurons, self.num_neurons))
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i == j:
                    jacobian[i, j] = (self.step_size / self.tau[i]) * (-1.0 + self.weights[i, j] * sigmoid_deriv[j])
                else:
                    jacobian[i, j] = (self.step_size / self.tau[i]) * self.weights[i, j] * sigmoid_deriv[j]
        
        return jacobian
    
    def get_lyapunov_exponent(
        self,
        perturbation_magnitude: float = 1e-6,
        timesteps: int = 10000,
        discard_transient: int = 1000,
        external_inputs: Optional[np.ndarray] = None
    ) -> float:
        """
        Estimate the largest Lyapunov exponent.
        
        Positive Lyapunov exponent indicates chaotic behavior, useful for understanding
        whether the neural network operates in a chaotic regime.
        
        Args:
            perturbation_magnitude: Size of initial perturbation.
            timesteps: Number of steps to integrate.
            discard_transient: Number of steps to discard before averaging.
            external_inputs: Constant external input (or None for zero input).
        
        Returns:
            Estimated largest Lyapunov exponent.
        """
        # Save original state
        original_state = self.state.copy()
        
        if external_inputs is None:
            external_inputs = np.zeros(self.num_neurons)
        
        lyapunov = 0.0
        count = 0
        
        # Run transient
        for _ in range(discard_transient):
            self.step(external_inputs)
        
        # Run and estimate exponent
        for _ in range(timesteps):
            # Create slightly perturbed system
            self.state += perturbation_magnitude * np.random.randn(self.num_neurons)
            
            # Run one step
            self.step(external_inputs)
            
            # Get distance in state space
            distance = np.linalg.norm(self.state)
            if distance > 0:
                lyapunov += np.log(distance / perturbation_magnitude)
                count += 1
            
            # Renormalize perturbation
            if distance > 0:
                self.state = self.state / distance * perturbation_magnitude
        
        # Restore original state
        self.state = original_state
        
        if count > 0:
            return lyapunov / (count * self.step_size)
        return 0.0
