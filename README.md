# Inference Engine

## Implemented Kernels
list of all kernels implemented and optimized for speed and performance

## Softmax
- used bli_dinvscalv function and plain loop to compute softmax of input x and outputs y in result
  
## Self Attention
- It includes the masking flag and is implemented without any if conditions eliminating branching in code

## Multi Attention
- This kernal is generic one, i.e. can be used for Multi Head Attention and Grouped Query attention.
- We have 2 parameters q_h for number of query heads and kv_h for number of K,V heads
- If q_h = kv_h, that means multi head attention otherwise grouped query attention
- This is implemented using mathematical calculations without ifs
  
## LayerNorm
- Benchmarked both implementation, blis vs plain c
- compared time of both approaches
- simple looping approach was around 3x faster so that implementation was used

## RMS Norm
- Benchmarked both implementation, blis vs plain c
- compared time of both approaches
- simple looping approach was around 2x faster so that implementation was used

## GELU
- calculate constants before loop
- in loop calculate output using formula and save it in output
  
## SILU
- plain c for loop with function computing output for each iterration
  
## RELU
- relu activation function implemented using masking to avoid branching and condition statements

## SIGMOID
- plain c for loop with function computing output for each iterration
  
## Testing approach
- For every kernal, load the c object file and matched the output with pytorch built-in functions
- Ran 1000 tests of random input for every kernal separately

## To Do List
- add asserts to validate inputs
- convert the independent loops into multi threading parallel processing
