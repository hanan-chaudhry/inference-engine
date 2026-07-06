## Kernals implemented so far
# Softmax
# Self attention
- It includes the masking flag and is implemented without any if
# multi Attention
- This kernal is generic one, i.e. can be used for Multi Head Attention and Grouped Query attention.
- We have 2 parameters q_h for number of query heads and kv_h for number of K,V heads
- If q_h = kv_h, that means multi head attention otherwise grouped query attention
- This is implemented using mathematical calculations without ifs
# LayerNorm
- Implemented with loops and blis operations
- compared time of both approaches
- simple looping approach was faster

## Testing approach
- For every kernal, load the c object file and matched the output with pytorch built-in functions
- Ran 1000 tests of random input for every kernal separately
## To Do in Future
- add asserts to validate inputs
- convert the independent loops into multi threading parallel processing
