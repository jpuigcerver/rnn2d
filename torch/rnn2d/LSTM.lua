local LSTM, parent = torch.class('rnn2d.LSTM', 'nn.Module')

LSTM.fw_inference = {
  ['torch.FloatTensor']      = rnn2d.cpu.rnn2d_lstm_cpu_float_fw_inference,
  ['torch.DoubleTensor']     = rnn2d.cpu.rnn2d_lstm_cpu_double_fw_inference,
  ['torch.CudaTensor']       = rnn2d.gpu.rnn2d_lstm_gpu_float_fw_inference,
  ['torch.CudaDoubleTensor'] = rnn2d.gpu.rnn2d_lstm_gpu_double_fw_inference
}

LSTM.fw_training = {
  ['torch.FloatTensor']      = rnn2d.cpu.rnn2d_lstm_cpu_float_fw_training,
  ['torch.DoubleTensor']     = rnn2d.cpu.rnn2d_lstm_cpu_double_fw_training,
  ['torch.CudaTensor']       = rnn2d.gpu.rnn2d_lstm_gpu_float_fw_training,
  ['torch.CudaDoubleTensor'] = rnn2d.gpu.rnn2d_lstm_gpu_double_fw_training
}

LSTM.bw_workspace = {
  ['torch.FloatTensor']      = rnn2d.cpu.rnn2d_lstm_cpu_float_bw_workspace,
  ['torch.DoubleTensor']     = rnn2d.cpu.rnn2d_lstm_cpu_double_bw_workspace,
  ['torch.CudaTensor']       = rnn2d.gpu.rnn2d_lstm_gpu_float_bw_workspace,
  ['torch.CudaDoubleTensor'] = rnn2d.gpu.rnn2d_lstm_gpu_double_bw_workspace
}

LSTM.bw_input = {
  ['torch.FloatTensor']      = rnn2d.cpu.rnn2d_lstm_cpu_float_bw_input,
  ['torch.DoubleTensor']     = rnn2d.cpu.rnn2d_lstm_cpu_double_bw_input,
  ['torch.CudaTensor']       = rnn2d.gpu.rnn2d_lstm_gpu_float_bw_input,
  ['torch.CudaDoubleTensor'] = rnn2d.gpu.rnn2d_lstm_gpu_double_bw_input
}

LSTM.bw_param = {
  ['torch.FloatTensor']      = rnn2d.cpu.rnn2d_lstm_cpu_float_bw_param,
  ['torch.DoubleTensor']     = rnn2d.cpu.rnn2d_lstm_cpu_double_bw_param,
  ['torch.CudaTensor']       = rnn2d.gpu.rnn2d_lstm_gpu_float_bw_param,
  ['torch.CudaDoubleTensor'] = rnn2d.gpu.rnn2d_lstm_gpu_double_bw_param
}

function LSTM:__init(inputSize, hiddenSize)
   parent.__init(self)
   assert(inputSize ~= nil)
   assert(hiddenSize ~= nil)

   self.inputSize = inputSize
   self.hiddenSize = hiddenSize

   self.weight = torch.Tensor():type(self:type())
   self.workspace = torch.Tensor():type(self:type())
   self.output = torch.Tensor():type(self:type())
   self.gradInput = torch.Tensor():type(self:type())
   self.gradWeight = torch.Tensor():type(self:type())

   self._backprop_workspace_done = false

   self:training()
   self:reset()
end

function LSTM:reset(stdv)
   stdv = stdv or 1.0 / math.sqrt(self.hiddenSize)
   local weightSize =
     4 * (1 + self.inputSize + 2 * self.hiddenSize) * 5 * self.hiddenSize
   self.weight:resize(weightSize)
   self.weight:uniform(-stdv, stdv)
   self.gradWeight:resizeAs(self.weight)
end

function LSTM:makeContiguous(input, gradOutput)
  if not input:isContiguous() or input:type() ~= self:type() then
    self._input = self._input or input.new():type(self:type())
    self._input = self._input:resizeAs(input):copy(input)
    input = self._input
  end
  if gradOutput and (not gradOutput:isContiguous() or
		     gradOutput:type() ~= self:type()) then
    self._gradOutput = self._gradOutput or gradOutput.new():type(self:type())
    self._gradOutput = self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
    gradOutput = self._gradOutput
  end
  return input, gradOutput
end

function LSTM:updateOutput(input)
  assert(input:dim() == 4, 'Input must have 4 dimensions: H x W x N x D')
  local H, W, N, K = input:size(1), input:size(2), input:size(3), input:size(4)
  local D = self.hiddenSize
  assert(self.inputSize == K, 'Incorrect input size!')
  self._backprop_workspace_done = false
  local x = self:makeContiguous(input)
  -- TODO(jpuigcerver): Use specialized functions for inference which require
  -- less computation and/or space
  local fw = LSTM.fw_training[self:type()]
  assert(fw ~= nil, ('Layer not implemented for type %q'):format(self:type()))
  self.output = self.output:resize(H, W, N, 4 * D)
  self.workspace = self.workspace:resize(self.output:nElement() * 2 * 6):zero()
  self.gradInput:zero()
  fw(H, W, N, K, D, torch.data(x), nil,	torch.data(self.weight),
     torch.data(self.output), torch.data(self.workspace))
  return self.output
end

function LSTM:backpropWorkspace(input, gradOutput)
  assert(self.train)
  assert(input:dim() == 4, 'Input must have 4 dimensions: H x W x N x D')
  local H, W, N, K = input:size(1), input:size(2), input:size(3), input:size(4)
  local D = self.hiddenSize
  assert(self.inputSize == K, 'Incorrect input size!')
  assert(gradOutput:isSameSizeAs(self.output),
	 'output and gradOutput sizes differ')
  local x, dy = self:makeContiguous(input, gradOutput)
  if not self._backprop_workspace_done then
    local bw = LSTM.bw_workspace[self:type()]
    assert(bw ~= nil, ('Layer not implemented for type %q'):format(self:type()))
    bw(H, W, N, K, D, torch.data(x), nil, torch.data(self.weight),
       torch.data(self.output), torch.data(dy), torch.data(self.workspace))
    self._backprop_workspace_done = true
  end
  return x
end

function LSTM:updateGradInput(input, gradOutput)
  local x = self:backpropWorkspace(input, gradOutput)
  local H, W, N, K = x:size(1), x:size(2), x:size(3), x:size(4)
  local D = self.hiddenSize
  local bw = LSTM.bw_input[self:type()]
  assert(bw ~= nil, ('Layer not implemented for type %q'):format(self:type()))
  if not self.gradInput:isSameSizeAs(x) then
    self.gradInput = self.gradInput:resizeAs(x):zero()
  end
  bw(H, W, N, K, D, torch.data(self.weight), 1.0, torch.data(self.gradInput),
     torch.data(self.workspace))
  return self.gradInput
end

function LSTM:accGradParameters(input, gradOutput, scale)
  scale = scale or 1
  local x = self:backpropWorkspace(input, gradOutput)
  local H, W, N, K = x:size(1), x:size(2), x:size(3), x:size(4)
  local D = self.hiddenSize
  local bw = LSTM.bw_param[self:type()]
  assert(bw ~= nil, ('Layer not implemented for type %q'):format(self:type()))
  bw(H, W, N, K, D, torch.data(x), torch.data(self.output), scale,
     torch.data(self.gradWeight), torch.data(self.workspace))
end

function LSTM:clearState()
  nn.utils.clear(self, '_input', '_gradOutput', 'workspace')
  return parent.clearState(self)
end

return LSTM
