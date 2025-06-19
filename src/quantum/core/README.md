# 🏗️ Strawberry Fields Architecture Deep Dive

## 🎓 **Lesson 2: Understanding the SF Computational Model**

Welcome to the heart of Strawberry Fields! This lesson will transform you from a user into an architect of quantum computing systems.

---

## 🧠 **The Big Picture: Why Programs vs Direct Execution?**

### **The Classical Approach (What You Might Expect):**
```python
# How you might think quantum computing works
quantum_state = create_vacuum()
quantum_state = apply_squeezing(quantum_state, 0.5)
quantum_state = apply_beamsplitter(quantum_state, π/4)
result = measure(quantum_state)
```

**Problem:** This is **immediate execution** - operations happen as you call them. But quantum computers are **expensive** and **noisy**. We need optimization!

### **The SF Approach (Program-Engine Model):**
```python
# How SF actually works
program = sf.Program(2)  # Create symbolic program

with program.context as q:
    ops.Sgate(0.5) | q[0]         # Symbolic operation
    ops.BSgate(π/4) | (q[0], q[1]) # Symbolic operation
    ops.MeasureX | q[0]           # Symbolic operation

# NOW execute everything optimally
engine = sf.Engine("tf")
result = engine.run(program)
```

**Advantage:** 
- **Optimization**: Engine can optimize the entire circuit before execution
- **Compilation**: Different backends (simulation, hardware) can compile differently
- **Reusability**: Same program, different parameters or backends
- **Debugging**: Inspect circuit before execution

---

## 🎭 **The Program: Your Quantum Script**

### **What is a Program?**

A `sf.Program` is like a **quantum assembly language script**:

```python
program = sf.Program(n_modes)  # Allocate n quantum modes

with program.context as q:
    # q[0], q[1], ..., q[n-1] are "quantum register references"
    # These are NOT actual quantum states yet!
    
    ops.Sgate(r) | q[0]           # "Apply squeezing to mode 0"
    ops.Dgate(α) | q[1]           # "Apply displacement to mode 1" 
    ops.BSgate(θ, φ) | (q[0], q[1])  # "Apply beam splitter to modes 0,1"
```

**Key Insight:** `q[0]` is **not a quantum state** - it's a **symbolic reference** to "mode 0 in this program".

### **The Context Manager Magic:**

```python
with program.context as q:
    # Inside: you're building a symbolic circuit
    ops.Sgate(0.5) | q[0]  # Adds operation to program's operation list
    
# Outside: program is "compiled" and ready to run
```

The context manager ensures operations are added to the program correctly.

### **Program Anatomy:**

```python
program = sf.Program(2)

# After building, program contains:
print(program.circuit)  # List of operations
# [Sgate(0.5) | (q[0],), BSgate(π/4, 0) | (q[0], q[1])]

print(program.parameters)  # Free parameters (more on this below)
print(program.num_modes)   # Number of quantum modes
```

---

## 🚀 **The Engine: Your Quantum Executor**

### **What is an Engine?**

An `sf.Engine` is a **quantum computer interface**:

```python
engine = sf.Engine(backend)
```

**Backend Options:**
- `"tf"`: TensorFlow backend (simulation, great for ML)
- `"fock"`: Fock basis simulation (exact, limited size)
- `"gaussian"`: Gaussian backend (efficient for Gaussian states)
- Hardware backends (real quantum computers!)

### **Execution Model:**

```python
# Execute program
result = engine.run(program, args=parameter_dict)

# Result contains:
result.state       # Final quantum state
result.samples     # Measurement results (if any)
result.ancillae    # Ancillary mode info
```

### **The args Parameter (Critical for ML!):**

```python
# Program with parameters
program = sf.Program(2)
with program.context as q:
    ops.Sgate(program.params('squeeze_r')) | q[0]  # Free parameter!
    ops.Dgate(program.params('displace_α')) | q[1]

# Execute with specific parameter values
result = engine.run(program, args={
    'squeeze_r': 0.5,
    'displace_α': 1.2 + 0.3j
})
```

This is how we connect **TensorFlow variables** to **quantum operations**!

---

## 🔗 **Parameter Management: The TF-SF Bridge**

### **The Challenge Recap:**

- **TensorFlow**: Wants `tf.Variable` objects for automatic differentiation
- **Strawberry Fields**: Wants scalar values for quantum operations
- **Goal**: Seamless integration with gradient flow

### **SF Parameter System:**

```python
# Method 1: Symbolic parameters
program = sf.Program(2)
with program.context as q:
    # Create symbolic parameter
    r_param = program.params('r')  # Returns a FreeParameter object
    ops.Sgate(r_param) | q[0]

# Method 2: Direct parameter creation
r_param = sf.FreeParameter('r')
ops.Sgate(r_param) | q[0]
```

### **The Bridge: tf.Variable → SF Parameter:**

```python
class QuantumLayer:
    def __init__(self):
        # TensorFlow side: trainable parameters
        self.tf_squeeze = tf.Variable(0.1, name='squeeze_param')
        self.tf_displace = tf.Variable(0.5 + 0.2j, name='displace_param')
        
        # SF side: symbolic program
        self.program = sf.Program(2)
        with self.program.context as q:
            ops.Sgate(self.program.params('r')) | q[0]
            ops.Dgate(self.program.params('α')) | q[1]
    
    def execute(self):
        # Bridge: TF Variables → SF parameters
        param_dict = {
            'r': self.tf_squeeze,      # TF Variable → SF parameter
            'α': self.tf_displace      # TF Variable → SF parameter
        }
        
        result = engine.run(self.program, args=param_dict)
        return result
```

**The Magic:** SF automatically handles TensorFlow tensors in the `args` dict!

---

## 🧮 **State Evolution and Backends**

### **TensorFlow Backend Deep Dive:**

The `"tf"` backend is special for machine learning:

```python
engine = sf.Engine("tf", backend_options={
    "cutoff_dim": 10,        # Fock space truncation
    "batch_size": None,      # Enable batching 
    "eval": True            # Return TF tensors (not numpy)
})
```

**Cutoff Dimension:**
- CV quantum states are infinite-dimensional
- Computers need finite representation
- `cutoff_dim=10` means we keep Fock states |0⟩, |1⟩, ..., |9⟩
- **Trade-off**: Higher cutoff = more accurate, more memory

**Batch Processing:**
```python
# Single sample
args = {'r': 0.5}  # Scalar

# Batch of samples  
args = {'r': tf.constant([0.5, 0.3, 0.8])}  # Vector
```

SF can process multiple parameter sets simultaneously!

### **State Representation:**

```python
result = engine.run(program)
state = result.state

# State methods:
state.fock_prob([n1, n2, ...])    # Probability of Fock state |n1,n2,...⟩
state.ket()                       # Full state vector
state.reduced_dm(modes)           # Density matrix for subset of modes
state.quad_expectation(mode, phi) # Quadrature expectation ⟨X_φ⟩
```

**For Machine Learning:**
```python
# Extract classical information
x_values = [state.quad_expectation(i, 0) for i in range(n_modes)]  # X quadratures
p_values = [state.quad_expectation(i, π/2) for i in range(n_modes)]  # P quadratures

measurements = tf.stack(x_values + p_values)  # Classical tensor for NN
```

---

## 🎯 **Advanced Topics: Performance and Optimization**

### **Memory Management:**

Quantum states grow exponentially:
```python
n_modes = 4
cutoff = 10
memory_needed = cutoff ** n_modes  # 10^4 = 10,000 complex numbers

# For larger systems:
n_modes = 8
cutoff = 6  
memory_needed = 6^8 = 1.6 million complex numbers ≈ 25 MB per state
```

**Strategy:** Careful cutoff selection based on application needs.

### **Compilation and Optimization:**

```python
# Engine optimization
engine = sf.Engine("tf", backend_options={
    "cutoff_dim": 6,
    "hbar": 2,              # Physical constant (affects squeezing)
    "pure": True,          # Use pure states (more efficient)
    "batch_size": 32       # Batch processing
})
```

### **Program Reuse Pattern:**

```python
class EfficientQuantumLayer:
    def __init__(self):
        # Build program once
        self.program = self._build_program()
        
        # Create engine once
        self.engine = sf.Engine("tf", backend_options={"cutoff_dim": 6})
    
    def _build_program(self):
        prog = sf.Program(self.n_modes)
        with prog.context as q:
            # Build symbolic circuit
            for i in range(self.n_modes):
                ops.Sgate(prog.params(f'squeeze_{i}')) | q[i]
            for i in range(self.n_modes - 1):
                ops.BSgate(prog.params(f'bs_theta_{i}'), 
                          prog.params(f'bs_phi_{i}')) | (q[i], q[i+1])
        return prog
    
    def execute(self, tf_parameters):
        # Convert TF parameters to arg dict
        args = self._tf_to_sf_args(tf_parameters)
        
        # Execute (program and engine already optimized)
        result = self.engine.run(self.program, args=args)
        return result
```

**Benefits:**
- Program compiled once
- Engine initialized once  
- Only parameter values change between calls

---

## 🔧 **Production Implementation Pattern**

Here's the pattern we'll use in our quantum GAN:

```python
class ProductionQuantumCircuit:
    """Production-ready SF quantum circuit with full TF integration"""
    
    def __init__(self, n_modes, n_layers, cutoff_dim=6):
        self.n_modes = n_modes
        self.n_layers = n_layers
        
        # TensorFlow parameters (trainable)
        self.tf_parameters = self._create_tf_parameters()
        
        # SF program (symbolic, created once)
        self.sf_program = self._build_sf_program()
        
        # SF engine (optimized, created once)
        self.sf_engine = sf.Engine("tf", backend_options={
            "cutoff_dim": cutoff_dim,
            "eval": True,  # Return TF tensors
            "batch_size": None  # Allow batching
        })
    
    def _create_tf_parameters(self):
        """Create TensorFlow variables for all quantum parameters"""
        params = {}
        
        for layer in range(self.n_layers):
            # Squeezing parameters
            for mode in range(self.n_modes):
                params[f'squeeze_{layer}_{mode}'] = tf.Variable(
                    tf.random.normal([1], stddev=0.1), name=f'squeeze_{layer}_{mode}'
                )
            
            # Beam splitter parameters  
            for mode in range(self.n_modes - 1):
                params[f'bs_theta_{layer}_{mode}'] = tf.Variable(
                    tf.random.normal([1], stddev=0.1), name=f'bs_theta_{layer}_{mode}'
                )
                params[f'bs_phi_{layer}_{mode}'] = tf.Variable(
                    tf.random.normal([1], stddev=0.1), name=f'bs_phi_{layer}_{mode}'
                )
        
        return params
    
    def _build_sf_program(self):
        """Build symbolic SF program with free parameters"""
        prog = sf.Program(self.n_modes)
        
        with prog.context as q:
            for layer in range(self.n_layers):
                # Squeezing layer
                for mode in range(self.n_modes):
                    ops.Sgate(prog.params(f'squeeze_{layer}_{mode}')) | q[mode]
                
                # Beam splitter layer (creates entanglement)
                for mode in range(self.n_modes - 1):
                    ops.BSgate(
                        prog.params(f'bs_theta_{layer}_{mode}'),
                        prog.params(f'bs_phi_{layer}_{mode}')
                    ) | (q[mode], q[mode + 1])
        
        return prog
    
    def execute(self, input_encoding=None):
        """Execute quantum circuit with current TF parameter values"""
        
        # Convert TF variables to SF args
        args = {}
        for name, tf_var in self.tf_parameters.items():
            args[name] = tf_var
        
        # Add input encoding if provided
        if input_encoding is not None:
            # Encode input as displacement operations
            prog_with_input = self._add_input_encoding(input_encoding)
            result = self.sf_engine.run(prog_with_input, args=args)
        else:
            result = self.sf_engine.run(self.sf_program, args=args)
        
        return result.state
    
    def extract_measurements(self, state):
        """Extract classical measurements from quantum state"""
        measurements = []
        
        for mode in range(self.n_modes):
            # X and P quadratures for each mode
            x_val = state.quad_expectation(mode, 0)      # X quadrature  
            p_val = state.quad_expectation(mode, π/2)    # P quadrature
            measurements.extend([x_val, p_val])
        
        return tf.stack(measurements)
    
    @property
    def trainable_variables(self):
        """Return list of trainable TF variables"""
        return list(self.tf_parameters.values())
```

---

## 🎓 **Key Takeaways for Quantum ML**

### **1. Separation of Concerns:**
- **Program**: Circuit structure (what operations, in what order)
- **Parameters**: Trainable values (TF variables)
- **Engine**: Execution strategy (simulation vs hardware)

### **2. Optimization Strategy:**
- Build programs once, execute many times
- Reuse engines across executions
- Batch processing for multiple samples

### **3. TF Integration:**
- SF automatically handles TF tensors in `args`
- Gradients flow through quantum operations
- State extraction provides classical interface

### **4. Memory Awareness:**
- Cutoff dimension controls accuracy vs memory
- Exponential scaling requires careful system design
- Pure states are more efficient than mixed states

---

## 🚀 **Next Steps**

Now you understand how SF works under the hood! Next, we'll dive into **quantum measurements** - how to extract maximum classical information from quantum states.

**Preview:** In `measurements/README.md`, you'll learn:
- Why X and P quadratures are optimal
- How to design measurement strategies
- Information-theoretic optimization of quantum-classical interfaces

---

## 🔗 **Navigation**

- **Previous:** `../README.md` - Quantum Computing Foundations
- **Next:** `../measurements/README.md` - Quantum Information Extraction  
- **Implementation:** `sf_production_circuit.py` - Production-ready implementation

---

**Remember:** SF's Program-Engine model enables the **optimization and reusability** that makes quantum machine learning practical. Master this pattern, and you'll build efficient, scalable quantum ML systems!
