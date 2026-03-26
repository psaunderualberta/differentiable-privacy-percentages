Refactor and rework the way in which datasets are loaded into memory, indexed, and used within the training pipeline.

Current behaviour:
- The entire dataset is loaded into memory, and stored in @environments/dp_params.py::DP_RL_Params class. 
- this DP_RL_Params object is passed to a the vmapped @environments/dp.py::train_with_noise function, possibly distributing the entire dataset across all devices. 
- Gradient usage is mitigated through `jax.checkpoint` on @environmments/dp.py::training_step. 
- For moderately sized datasets, such as CIFAR-10, the memory usage for computing gradients across an entire model's training is too expensive, and JAX is unable to allocate enough memory even across 4 GPUs. 
- For even larger datasets like EyePACS, the dataset can't be loaded into memory to begin with. 

Desired behaviour:
- At most, only one copy of the dataset shall be stored across all devices. That is, no replication of data. 
- Ideally, only the parts of the dataset needed at any iteration of `train_with_noise`'s main scan will be loaded into the device's memory at any time. This might not be possible within a Jax-jitted function. 
- A balance should be struck between memory usage and execution time for @environments/outer_loop.py::get_policy_loss`. Reducing memory usage beyond reasonable levels is not acceptable if it results in a drastic runtime increase. 
- In addition to the actual dataset, the gradients backpropagated through the `train_with_noise` function will likely also need to be stored or managed to keep the memory usage low. Currently, that is done via `jax.checkpoint`, though that may not be sufficient with larger datasets. 

Notes:
- get_policy_loss, which calls the vmapped train_with_noise, must remain jittable for rapid training of machine learning models. 
- Sharding of models or datasets across GPUs is desirable, but not strictly necessary. However, it is near-certain that a solution to the desired behaviour will make use of array sharding. 
- Data is stored numpy files underneath the `src/data/<dataset>` folder, with separate files for X / y and training / validation datasets. Assume these files are in the correct format to be passed to a correctly-defined machine learning models, i.e. `jax.vmap(model)(jnp.asarray(np.load(X-datafile-train)))` will work as expected. 

Start two parallel agents to brainstorm possible solutions and plans. Do not implement code.
