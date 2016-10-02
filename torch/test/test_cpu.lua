require 'rnn2d'


local rnn2dtest = torch.TestSuite()
local jac = nn.Jacobian

local function testLayer(layer, input, gradOutput)
  local res = {}
  res.output = layer:forward(cast(input))
  layer:zeroGradParameters()
  res.gradInput = layer:backward(cast(input), cast(gradOutput), scale)
end

local mytester = torch.Tester()
mytester:add(rnn2dtest)
mytester:run()

local H, W, N, K, D = 2, 3, 2, 3, 2
local lstm2d = rnn2d.LSTM(K, D)
lstm2d.weight = torch.DoubleTensor({
    -- TOP-LEFT DIRECTION
    -- Bias
    -0.66, -0.56, -0.19,  0.94, -0.22,  0.12, -0.99, -0.08, -0.79, -0.69,
    -- Input weights
    -0.69,  0.16,  0.64,  0.11,  0.72, -0.48,  0.67,  0.72,  0.61, -0.34,
     0.72, -0.39, -0.31, -0.76,  0.31, -0.88,  0.24, -0.5,  -0.65, -0.21,
     0.61,  0.95, -0.46,  0.79,  0.98, -0.89,  0.88,  0.62, -0.36,  0.07,
    -- Recurrent weights in y-dimension
     0.16,  0.10,  0.01,  0.91, -0.05,  0.38,  0.38, -0.62,  0.99, -0.03,
     0.60,  0.30, -0.47, -0.03,  0.12, -0.77,  0.94,  0.77, -0.79,  0.76,
    -- Recurrent weights in x-dimension
    -0.30, -0.80,  0.93,  0.90,  0.95, -0.50,  0.65,  0.23, -0.90,  0.36,
    -0.42,  0.39,  0.54, -0.20,  0.14, -0.16,  0.57,  0.51, -0.30,  0.88,
    -- TOP-RIGHT DIRECTION
    -- Bias
    -0.66, -0.56, -0.19,  0.94, -0.22,  0.12, -0.99, -0.08, -0.79, -0.69,
    -- Input weights
    -0.69,  0.16,  0.64,  0.11,  0.72, -0.48,  0.67,  0.72,  0.61, -0.34,
     0.72, -0.39, -0.31, -0.76,  0.31, -0.88,  0.24, -0.5,  -0.65, -0.21,
     0.61,  0.95, -0.46,  0.79,  0.98, -0.89,  0.88,  0.62, -0.36,  0.07,
    -- Recurrent weights in y-dimension
     0.16,  0.10,  0.01,  0.91, -0.05,  0.38,  0.38, -0.62,  0.99, -0.03,
     0.60,  0.30, -0.47, -0.03,  0.12, -0.77,  0.94,  0.77, -0.79,  0.76,
    -- Recurrent weights in x-dimension
    -0.30, -0.80,  0.93,  0.90,  0.95, -0.50,  0.65,  0.23, -0.90,  0.36,
    -0.42,  0.39,  0.54, -0.20,  0.14, -0.16,  0.57,  0.51, -0.30,  0.88,
    -- BOTTOM-LEFT DIRECTION
    -- Bias
    -0.66, -0.56, -0.19,  0.94, -0.22,  0.12, -0.99, -0.08, -0.79, -0.69,
    -- Input weights
    -0.69,  0.16,  0.64,  0.11,  0.72, -0.48,  0.67,  0.72,  0.61, -0.34,
     0.72, -0.39, -0.31, -0.76,  0.31, -0.88,  0.24, -0.5,  -0.65, -0.21,
     0.61,  0.95, -0.46,  0.79,  0.98, -0.89,  0.88,  0.62, -0.36,  0.07,
    -- Recurrent weights in y-dimension
     0.16,  0.10,  0.01,  0.91, -0.05,  0.38,  0.38, -0.62,  0.99, -0.03,
     0.60,  0.30, -0.47, -0.03,  0.12, -0.77,  0.94,  0.77, -0.79,  0.76,
    -- Recurrent weights in x-dimension
    -0.30, -0.80,  0.93,  0.90,  0.95, -0.50,  0.65,  0.23, -0.90,  0.36,
    -0.42,  0.39,  0.54, -0.20,  0.14, -0.16,  0.57,  0.51, -0.30,  0.88,
    -- BOTTOM-RIGHT DIRECTION
    -- Bias
    -0.66, -0.56, -0.19,  0.94, -0.22,  0.12, -0.99, -0.08, -0.79, -0.69,
    -- Input weights
    -0.69,  0.16,  0.64,  0.11,  0.72, -0.48,  0.67,  0.72,  0.61, -0.34,
     0.72, -0.39, -0.31, -0.76,  0.31, -0.88,  0.24, -0.5,  -0.65, -0.21,
     0.61,  0.95, -0.46,  0.79,  0.98, -0.89,  0.88,  0.62, -0.36,  0.07,
    -- Recurrent weights in y-dimension
     0.16,  0.10,  0.01,  0.91, -0.05,  0.38,  0.38, -0.62,  0.99, -0.03,
     0.60,  0.30, -0.47, -0.03,  0.12, -0.77,  0.94,  0.77, -0.79,  0.76,
    -- Recurrent weights in x-dimension
    -0.30, -0.80,  0.93,  0.90,  0.95, -0.50,  0.65,  0.23, -0.90,  0.36,
    -0.42,  0.39,  0.54, -0.20,  0.14, -0.16,  0.57,  0.51, -0.30,  0.88})

local input = torch.DoubleTensor({
    0.30, 0.68, 0.29, 0.10, 0.70, 0.88, 0.13, 0.18, 0.35, 0.86, 0.66, 0.75,
    0.53, 0.40, 0.48, 0.20, 0.58, 0.66, 0.30, 0.99, 0.64, 0.46, 0.44, 0.65,
    0.82, 0.59, 0.47, 0.18, 0.53, 0.13, 0.68, 0.79, 0.80, 0.32, 0.09, 0.40})
input:resize(H, W, N, K)

--local input = torch.FloatTensor(2, 3, 2, 3):uniform(-1, 1)
--local gradOutput = torch.FloatTensor(8, 9, 10, 3 * 4):uniform(-1, 2)

local gradOutput = torch.DoubleTensor({
    0.51,  0.10,  0.21,  0.47, -0.06,  0.26,  0.50, -0.71,  0.53,  0.65,
    0.52,  0.25, -0.39, -0.13,  0.05,  0.07,  0.44,  0.66,  0.30,  0.98,
    0.20,  0.76, -0.93,  0.42,  0.17,  0.71,  0.16, -0.48,  0.39,  0.92,
    0.04,  0.81,  0.07,  0.98, -0.17,  0.79,  0.57,  0.39,  0.94,  0.40,
    0.81,  0.40,  0.81,  0.34,  0.74,  0.49,  0.68,  0.00,  0.29,  0.29,
    0.50,  0.52, -0.15, -0.63, -0.87,  0.43,  0.39,  0.59, -0.68,  0.92,
    0.43, -0.16, -0.27,  0.19, -0.84,  0.13,  0.33,  0.89, -0.47,  0.72,
   -0.47,  0.27,  0.85, -0.23,  0.15, -0.61,  0.69,  0.76,  0.47,  0.56,
    0.13,  0.61,  0.71,  0.11, -0.44,  0.11,  0.47,  0.04, -0.34,  0.78,
    0.80,  0.24,  0.40,  0.49, -0.93,  0.09})
gradOutput:resize(H, W, N, 4 * D)


input = input:float()
gradOutput = gradOutput:float()
lstm2d:float()
lstm2d:zeroGradParameters()

--print(jac.testJacobian(lstm2d, input))
local output = lstm2d:forward(input)
local gradInput = lstm2d:backward(input, gradOutput)

print(string.format('sum_I  = %.18f', input:sum()))
print(string.format('sum_P  = %.18f', lstm2d.weight:sum()))
print(string.format('sum_dO = %.18f', gradOutput:sum()))
print(string.format('sum_Q  = %.18f', lstm2d.workspace:sum()))
print(string.format('sum_O  = %.18f', output:sum()))
print(string.format('sum_dQ = %.18f', lstm2d.gradWorkspace:sum()))
print(string.format('sum_dI = %.18f', gradInput:sum()))
print(string.format('sum_dP = %.18f', lstm2d.gradWeight:sum()))


os.exit(1)
for y=1,H do
  for x=1,W do
    io.write(string.format('O(%d,%d,.,.) =\n', y, x))
    for n=1,N do
      for d=1,4*D do
	io.write(string.format(' % .4f', output[{y,x,n,d}]))
      end
      io.write('\n')
    end
    io.write('\n')
  end
end
