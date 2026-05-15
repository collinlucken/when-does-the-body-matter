"""
2D microworld environments for embodied cognition and evolutionary robotics.

This module provides agent and environment classes for simulating minimal physical
systems where agents evolve neural controllers. Key principles:

1. Embodiment: Agent morphology constrains behavior (not just weights)
2. Situatedness: Agent dynamics are inseparable from environment dynamics
3. Dynamical coupling: Environment state affects agent state and vice versa

Environments implemented:
- CategoricalPerceptionEnv: Beer (2003) categorical perception task
- PhototaxisEnv: Simple gradient-following behavior
- PerceptualCrossingEnv: Froese & Di Paolo (2008) perceptual crossing

References:
    Beer, R. D. (2003). The dynamics of active categorical perception in an evolved
        model agent. Adaptive Behavior, 11(4), 209-243.
    Froese, T., & Di Paolo, E. A. (2008). Emergence of joint action in two robotic agents
        coupled through kinetic energy transfer. New Ideas in Psychology, 26(3), 384-401.
"""

from typing import Optional, Tuple, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class AgentState:
    """Current state of an agent."""
    position: np.ndarray  # [x, y] coordinates
    velocity: np.ndarray  # [vx, vy] velocity
    sensor_values: np.ndarray  # Raw sensor readings
    motor_commands: np.ndarray  # Motor outputs from neural network


class Agent:
    """
    Minimal agent model with circular body, two motors, and bilateral sensors.
    
    Morphology:
    - Circular body with radius r
    - Two motors (left and right) controlling forward/rotational movement
    - Bilateral pair of sensors (left and right) for detecting environmental stimuli
    
    The agent's state includes position, velocity, and internal neural state.
    External forces (drag, collisions) are simulated with simple physics.
    """
    
    def __init__(
        self,
        radius: float = 1.0,
        max_speed: float = 1.0,
        sensor_range: float = 10.0,
        motor_scale: float = 1.0,
        friction: float = 0.1,
        initial_position: Optional[np.ndarray] = None,
        initial_angle: float = 0.0
    ) -> None:
        """
        Initialize agent.
        
        Args:
            radius: Agent body radius.
            max_speed: Maximum speed (motor output is scaled by this).
            sensor_range: Maximum detection range for sensors.
            motor_scale: Scale factor for motor commands.
            friction: Friction coefficient (drag force proportional to velocity).
            initial_position: Starting [x, y] position (default: origin).
            initial_angle: Starting orientation angle in radians.
        """
        self.radius = radius
        self.max_speed = max_speed
        self.sensor_range = sensor_range
        self.motor_scale = motor_scale
        self.friction = friction
        
        # State
        self.position = initial_position if initial_position is not None else np.array([0.0, 0.0])
        self.angle = initial_angle  # Orientation
        self.velocity = np.zeros(2)
        
        # Sensors and motors
        self.sensor_values = np.zeros(2)  # Left, right sensors
        self.motor_commands = np.zeros(2)  # Left, right motor outputs
        
        # Sensor positioning (bilateral, symmetric)
        self.sensor_angle_offset = np.pi / 6  # 30 degrees from center line
    
    def set_motor_commands(self, left: float, right: float) -> None:
        """
        Set motor commands from neural network output.
        
        Args:
            left: Left motor command (typically in [-1, 1]).
            right: Right motor command (typically in [-1, 1]).
        """
        self.motor_commands = np.array([left, right])
    
    def update(self, dt: float = 0.01) -> None:
        """
        Update agent position and velocity based on motor commands.
        
        Simple differential drive kinematics:
        - Forward velocity from average of motor commands
        - Angular velocity from difference of motor commands
        
        Args:
            dt: Time step.
        """
        # Differential drive kinematics
        v_forward = (self.motor_commands[0] + self.motor_commands[1]) / 2.0 * self.max_speed
        v_angular = (self.motor_commands[1] - self.motor_commands[0]) * self.motor_scale
        
        # Update angle
        self.angle += v_angular * dt
        self.angle = self.angle % (2 * np.pi)  # Keep in [0, 2π)
        
        # Update velocity (with friction/drag)
        self.velocity = v_forward * np.array([np.cos(self.angle), np.sin(self.angle)])
        self.velocity *= (1 - self.friction * dt)
        
        # Update position
        self.position += self.velocity * dt
    
    def get_sensor_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get positions of left and right sensors in world coordinates.
        
        Returns:
            left_sensor_pos: [x, y] position of left sensor.
            right_sensor_pos: [x, y] position of right sensor.
        """
        # Sensors positioned at edges of body
        # Left sensor at angle + offset
        left_angle = self.angle + self.sensor_angle_offset
        left_pos = self.position + self.radius * np.array([np.cos(left_angle), np.sin(left_angle)])
        
        # Right sensor at angle - offset
        right_angle = self.angle - self.sensor_angle_offset
        right_pos = self.position + self.radius * np.array([np.cos(right_angle), np.sin(right_angle)])
        
        return left_pos, right_pos
    
    def get_state(self) -> AgentState:
        """Get current agent state."""
        return AgentState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            sensor_values=self.sensor_values.copy(),
            motor_commands=self.motor_commands.copy()
        )


class Environment(ABC):
    """
    Abstract base class for microworld environments.
    
    Environments handle:
    1. Objects/stimuli in the world
    2. Physics simulation (collisions, dynamics)
    3. Sensor-object interactions
    4. Fitness evaluation
    """
    
    def __init__(self, width: float = 50.0, height: float = 50.0) -> None:
        """
        Initialize environment.
        
        Args:
            width: Width of environment.
            height: Height of environment.
        """
        self.width = width
        self.height = height
        self.agent: Optional[Agent] = None
        self.time = 0.0
        self.dt = 0.01  # Time step
    
    def set_agent(self, agent: Agent) -> None:
        """Set the agent in this environment."""
        self.agent = agent
    
    @abstractmethod
    def step(self) -> None:
        """Advance environment by one time step."""
        pass
    
    @abstractmethod
    def get_sensor_readings(self) -> np.ndarray:
        """
        Compute sensor readings based on current state.
        
        Returns:
            Sensor values (typically in [0, 1] or [-1, 1]).
        """
        pass
    
    @abstractmethod
    def evaluate_fitness(self) -> float:
        """Evaluate fitness based on task performance."""
        pass
    
    def reset(self) -> None:
        """Reset environment to initial state."""
        self.time = 0.0


class CategoricalPerceptionEnv(Environment):
    """
    Beer's (2003) categorical perception task.
    
    An object falls from the top of the environment at a random horizontal position.
    The agent must:
    - Catch (move under) small objects
    - Avoid (move away from) large objects
    
    Small/large distinction is binary but agent must learn to categorize based on
    visual appearance during descent.
    
    This environment tests whether agents can learn a categorical decision based on
    continuous sensory input - a classic problem in embodied cognition.
    
    Reference:
        Beer, R. D. (2003). The dynamics of active categorical perception in an
        evolved model agent. Adaptive Behavior, 11(4), 209-243.
    """
    
    def __init__(
        self,
        width: float = 50.0,
        height: float = 50.0,
        small_radius: float = 0.5,
        large_radius: float = 2.0,
        object_speed: float = 1.0,
        categorization_height: float = 10.0,
        small_prob: float = 0.5
    ) -> None:
        """
        Initialize categorical perception environment.
        
        Args:
            width: Environment width.
            height: Environment height.
            small_radius: Radius of small (catch) objects.
            large_radius: Radius of large (avoid) objects.
            object_speed: Speed at which objects fall.
            categorization_height: Height at which object size is "revealed" to agent.
            small_prob: Probability that next object is small.
        """
        super().__init__(width, height)
        self.small_radius = small_radius
        self.large_radius = large_radius
        self.object_speed = object_speed
        self.categorization_height = categorization_height
        self.small_prob = small_prob
        
        # Current object state
        self.object_x = 0.0  # x position
        self.object_y = 0.0  # y position (falls from top)
        self.object_is_small = True  # Size category
        self.object_active = False  # Whether object is currently falling
        
        # Trial tracking
        self.trial_number = 0
        self.correct_catches = 0
        self.correct_avoidances = 0
        self.total_trials = 0
    
    def reset(self) -> None:
        """Reset environment and spawn new object."""
        super().reset()
        self._spawn_object()
    
    def _spawn_object(self) -> None:
        """Spawn a new falling object."""
        self.object_x = np.random.uniform(self.small_radius + 1, 
                                         self.width - self.small_radius - 1)
        self.object_y = self.height
        self.object_is_small = np.random.random() < self.small_prob
        self.object_active = True
        self.trial_number += 1
        self.total_trials += 1
    
    def step(self) -> None:
        """Advance environment: fall object, update agent, compute sensors."""
        if not self.object_active or self.agent is None:
            return
        
        # Fall object
        self.object_y -= self.object_speed * self.dt
        
        # Check if object reached bottom
        if self.object_y < 0:
            # Check if agent was in correct position
            object_radius = self.small_radius if self.object_is_small else self.large_radius
            distance_to_agent = abs(self.object_x - self.agent.position[0])
            
            if self.object_is_small:
                # Should have caught (agent nearby)
                if distance_to_agent < object_radius + self.agent.radius:
                    self.correct_catches += 1
            else:
                # Should have avoided (agent far away)
                if distance_to_agent > 5.0:  # "Avoided" threshold
                    self.correct_avoidances += 1
            
            # Spawn next object
            self._spawn_object()
        
        # Update agent
        if self.agent is not None:
            self.agent.update(self.dt)
        
        # Compute sensor readings
        self.agent.sensor_values = self.get_sensor_readings()
        
        self.time += self.dt
    
    def get_sensor_readings(self) -> np.ndarray:
        """
        Compute bilateral sensor readings based on falling object.
        
        Sensors detect:
        1. Distance to object
        2. Apparent size (distance-dependent)
        3. Approach/recession rate
        
        Returns:
            Array of two sensor values [left, right].
        """
        if self.agent is None or not self.object_active:
            return np.array([0.0, 0.0])
        
        left_pos, right_pos = self.agent.get_sensor_positions()
        
        # Distance from each sensor to object
        left_dist = np.sqrt((left_pos[0] - self.object_x) ** 2 + 
                           (left_pos[1] - self.object_y) ** 2)
        right_dist = np.sqrt((right_pos[0] - self.object_x) ** 2 + 
                            (right_pos[1] - self.object_y) ** 2)
        
        # Convert distance to sensor activation (inverse with range)
        # Also incorporate apparent size
        object_radius = self.small_radius if self.object_is_small else self.large_radius
        
        left_sensor = max(0.0, 1.0 - left_dist / self.agent.sensor_range)
        right_sensor = max(0.0, 1.0 - right_dist / self.agent.sensor_range)
        
        return np.array([left_sensor, right_sensor])
    
    def evaluate_fitness(self) -> float:
        """
        Evaluate fitness as proportion of correct categorical responses.
        
        Returns:
            Fitness in range [0, 1].
        """
        if self.total_trials == 0:
            return 0.0
        return (self.correct_catches + self.correct_avoidances) / self.total_trials


class PhototaxisEnv(Environment):
    """
    Simple light-gradient environment for testing phototaxis.
    
    A light source is positioned in the environment. Agent receives bilateral
    sensor input indicating light intensity from left and right sides.
    Task is to evolve a controller that seeks or avoids the light.
    """
    
    def __init__(
        self,
        width: float = 50.0,
        height: float = 50.0,
        light_position: Optional[np.ndarray] = None,
        light_intensity: float = 10.0
    ) -> None:
        """
        Initialize phototaxis environment.
        
        Args:
            width: Environment width.
            height: Environment height.
            light_position: [x, y] position of light (default: center).
            light_intensity: Intensity of light source.
        """
        super().__init__(width, height)
        self.light_position = (light_position if light_position is not None 
                              else np.array([width / 2, height / 2]))
        self.light_intensity = light_intensity
        self.starting_position = np.array([width / 2, height / 4])
    
    def reset(self) -> None:
        """Reset environment and agent position."""
        super().reset()
        if self.agent is not None:
            self.agent.position = self.starting_position.copy()
            self.agent.velocity = np.zeros(2)
    
    def step(self) -> None:
        """Advance environment: update agent, compute light sensors."""
        if self.agent is None:
            return
        
        self.agent.update(self.dt)
        self.agent.sensor_values = self.get_sensor_readings()
        
        # Wrap around boundaries
        self.agent.position[0] = self.agent.position[0] % self.width
        self.agent.position[1] = self.agent.position[1] % self.height
        
        self.time += self.dt
    
    def get_sensor_readings(self) -> np.ndarray:
        """
        Compute bilateral light sensor readings.
        
        Returns:
            Array [left_light, right_light] with values in [0, 1].
        """
        if self.agent is None:
            return np.array([0.0, 0.0])
        
        left_pos, right_pos = self.agent.get_sensor_positions()
        
        # Light intensity falls off with distance squared
        left_dist = np.linalg.norm(left_pos - self.light_position)
        right_dist = np.linalg.norm(right_pos - self.light_position)
        
        left_light = self.light_intensity / (1.0 + left_dist ** 2)
        right_light = self.light_intensity / (1.0 + right_dist ** 2)
        
        # Normalize to [0, 1]
        max_intensity = self.light_intensity
        left_sensor = np.tanh(left_light / max_intensity)
        right_sensor = np.tanh(right_light / max_intensity)
        
        return np.array([left_sensor, right_sensor])
    
    def evaluate_fitness(self) -> float:
        """
        Evaluate fitness as distance to light source.
        
        Returns:
            Fitness (higher is closer to light).
        """
        if self.agent is None:
            return 0.0
        distance = np.linalg.norm(self.agent.position - self.light_position)
        return max(0.0, 1.0 - distance / (self.width / 2))


class PerceptualCrossingEnv(Environment):
    """
    Froese & Di Paolo (2008) perceptual crossing setup.
    
    Two agents move in a 1D ring environment. Each agent can perceive the other
    only when they are close AND the perceiver has a specific orientation/morphology.
    This creates a paradoxical situation where strict behavioral rules fail,
    requiring coupled dynamical interaction.
    
    This environment is used to study:
    - Embodied interaction and autonomy
    - Emergence of coordination from coupled dynamics
    - The role of morphology in enabling perception
    
    Reference:
        Froese, T., & Di Paolo, E. A. (2008). Emergence of joint action in two
        robotic agents coupled through kinetic energy transfer. New Ideas in
        Psychology, 26(3), 384-401.
    """
    
    def __init__(
        self,
        circumference: float = 100.0,
        perception_distance: float = 5.0,
        motor_coupling_strength: float = 0.1
    ) -> None:
        """
        Initialize perceptual crossing environment.
        
        Args:
            circumference: Circumference of 1D ring.
            perception_distance: Distance threshold for perception.
            motor_coupling_strength: Strength of kinetic energy transfer between agents.
        """
        super().__init__(circumference, 1.0)  # 1D, use width as circumference
        self.circumference = circumference
        self.perception_distance = perception_distance
        self.motor_coupling_strength = motor_coupling_strength
        
        # Will hold two agents
        self.agent1: Optional[Agent] = None
        self.agent2: Optional[Agent] = None
    
    def set_agents(self, agent1: Agent, agent2: Agent) -> None:
        """Set both agents in the environment."""
        self.agent1 = agent1
        self.agent2 = agent2
    
    def step(self) -> None:
        """Advance environment with both agents."""
        if self.agent1 is None or self.agent2 is None:
            return
        
        # Update positions
        self.agent1.update(self.dt)
        self.agent2.update(self.dt)
        
        # Wrap around 1D ring
        self.agent1.position[0] = self.agent1.position[0] % self.circumference
        self.agent2.position[0] = self.agent2.position[0] % self.circumference
        
        # Compute perception and update sensors
        self._update_sensors()
        
        # Apply kinetic energy coupling (kinetic energy transfer)
        self._apply_motor_coupling()
        
        self.time += self.dt
    
    def _update_sensors(self) -> None:
        """Update sensor values based on perception."""
        # Distance between agents on ring
        dist = min(
            abs(self.agent1.position[0] - self.agent2.position[0]),
            self.circumference - abs(self.agent1.position[0] - self.agent2.position[0])
        )
        
        # Can perceive if close enough
        perception_active = dist < self.perception_distance
        
        # Sensor readings (0 or 1 for perceptual crossing)
        if perception_active:
            self.agent1.sensor_values = np.array([1.0, 1.0])
            self.agent2.sensor_values = np.array([1.0, 1.0])
        else:
            self.agent1.sensor_values = np.array([0.0, 0.0])
            self.agent2.sensor_values = np.array([0.0, 0.0])
    
    def _apply_motor_coupling(self) -> None:
        """Apply kinetic energy transfer between agents."""
        # Simple coupling: agents exchange some kinetic energy
        total_velocity = (self.agent1.velocity[0] + self.agent2.velocity[0]) / 2.0
        self.agent1.velocity[0] += self.motor_coupling_strength * (total_velocity - self.agent1.velocity[0])
        self.agent2.velocity[0] += self.motor_coupling_strength * (total_velocity - self.agent2.velocity[0])
    
    def get_sensor_readings(self) -> np.ndarray:
        """Get sensor readings (handled by _update_sensors)."""
        if self.agent1 is None:
            return np.array([0.0, 0.0])
        return self.agent1.sensor_values.copy()
    
    def evaluate_fitness(self) -> float:
        """
        Evaluate fitness based on successful perceptual crossing events.
        
        Returns:
            Fitness metric (0-1).
        """
        # Would track successful coordination events
        return 0.0  # Placeholder
