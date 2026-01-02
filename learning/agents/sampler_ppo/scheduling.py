import jax
import jax.numpy as jnp
import flax
from typing import Any, List, Tuple

# 1. Define the State
# This holds the changing data (weights, buffer, last choice).
@flax.struct.dataclass
class SchedulerState:
    log_weights: jnp.ndarray
    error_buffer: jnp.ndarray
    selected_arm: int
    prev_cvar: float
    step_count: int = 0
# 2. Define the Scheduler (The Logic/Config)
# This holds static configuration and pure functions.
@flax.struct.dataclass
class GMMScheduler:
    arms: jnp.ndarray
    lr: float = 0.2
    window_size: int = 5
    decay: float = 0.9
    use_std: bool = True
    epsilon: float = 1e-6

    @classmethod
    def create(cls, 
               arms: List[float] = [-1.0, 0.0, 1.0], 
               lr: float = 0.2, 
               window_size: int = 5, 
               decay: float = 0.9,
               use_std: bool = True) -> Tuple['GMMScheduler', SchedulerState]:
        """Factory method to initialize Scheduler and State."""
        
        # Convert arms to JAX array
        arms_array = jnp.array(arms)
        
        # Initialize Logic Container
        scheduler = cls(
            arms=arms_array,
            lr=lr,
            window_size=window_size,
            decay=decay,
            use_std=use_std
        )
        
        # Initialize State Container
        # We initialize log_weights to 0.0 (equal probability: exp(0)=1)
        init_state = SchedulerState(
            log_weights=jnp.zeros(len(arms)), 
            error_buffer=jnp.zeros(window_size),
            selected_arm=0,
            step_count=0,
            prev_cvar=0.,
        )
        
        return scheduler, init_state

    def get_probs(self, state: SchedulerState) -> jnp.ndarray:
        """Get probabilities for each arm (softmax of log_weights)."""
        return jax.nn.softmax(state.log_weights)

    def sample(self, state: SchedulerState, key: jax.random.PRNGKey) -> Tuple[SchedulerState, float]:
        """Sample an arm and return the new state (with selected arm recorded) and the arm value."""
        probs = self.get_probs(state)
        
        # Sample index based on probabilities
        arm_index = jax.random.categorical(key, jnp.log(probs + self.epsilon))
        
        # Get actual arm value (e.g., -1, 0, or 1)
        arm_value = self.arms[arm_index]
        
        # Return updated state with the selected arm recorded
        new_state = state.replace(selected_arm=arm_index)
        return new_state, arm_value

    def update_dists(self, state: SchedulerState, feedback: float) -> SchedulerState:
        """Update the distribution based on feedback."""
        
        # 1. Update the Error Buffer (Circular Buffer)
        # Roll buffer left, put new feedback at the end
        new_buffer = jnp.roll(state.error_buffer, -1)
        new_buffer = new_buffer.at[-1].set(feedback)
        
        # 2. Normalize Feedback
        # Calculate stats only on valid filled steps if needed, 
        # but for simplicity assuming buffer fills quickly or 0 initialization is acceptable.
        buffer_mean = jnp.mean(new_buffer)
        buffer_std = jnp.std(new_buffer)
        
        # Center the feedback
        norm_feedback = feedback - buffer_mean
        
        # Scale by std if requested
        if self.use_std:
            norm_feedback = norm_feedback / (buffer_std + self.epsilon)

        # 3. Calculate Weight Update
        # IPS (Inverse Propensity Scoring) Scaling: Feedback / Probability
        probs = self.get_probs(state)
        selected_prob = probs[state.selected_arm]
        
        # Gradient approximation
        grad = norm_feedback / (selected_prob + self.epsilon)
        
        # 4. Apply Update
        new_log_weights = state.log_weights
        
        # Apply decay to the selected arm's weight
        current_weight = new_log_weights[state.selected_arm]
        updated_weight = (current_weight * self.decay) + (self.lr * grad)
        
        new_log_weights = new_log_weights.at[state.selected_arm].set(updated_weight)
        
        return state.replace(
            log_weights=new_log_weights,
            error_buffer=new_buffer,
            step_count=state.step_count + 1
        )

# --- Usage Example ---

def main():
    key = jax.random.PRNGKey(5)
    
    # Initialize
    scheduler, state = GMMScheduler.create(
        arms=jnp.arange(61)-30, 
        lr=0.1, 
        window_size=5
    )
    
    # Mock Training Loop
    # Because we used flax.struct.dataclass, we can JIT the step!
    @jax.jit
    def train_step(k, s):
        k, subk = jax.random.split(k)
        
        # 1. Sample
        s, arm_val = scheduler.sample(s, subk)
        
        # 2. Mock Environment/Loss (Simulate receiving a "reward/error")
        # Let's pretend arm '0' is the best (lowest error)
        mock_feedback = -(arm_val + 15) ** 2 
        
        # 3. Update
        s = scheduler.update_dists(s, mock_feedback)
        return k, s, arm_val, mock_feedback

    print(f"Initial Probs: {scheduler.get_probs(state)}")
    training_steps = 100

    # Run a few steps
    for i in range(training_steps):
        key, state, arm, fb = train_step(key, state)
        print(f"Step {i}: Arm {arm:.1f} | Feedback {fb:.4f} | Probs {scheduler.get_probs(state)}")

if __name__ == "__main__":
    main()